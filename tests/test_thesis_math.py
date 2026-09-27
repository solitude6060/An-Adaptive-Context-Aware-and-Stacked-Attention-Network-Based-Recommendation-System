"""History-axis attention and stable pairwise log loss.

Reads main.py with ast and never imports the training CLI.
CPU runs replace Tensor.cuda only inside this file.
"""

import ast
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import stacked_AttRec  # noqa: E402


@contextmanager
def legacy_cuda_stays_on_cpu():
    original = torch.Tensor.cuda

    def _stay(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _stay
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def make_model(num_users, num_items, latent_dim, stack_num, head_num, time_info_mode, window_size):
    args = type(
        "Args",
        (),
        {"num_users": num_users, "num_items": num_items, "latent_dim": latent_dim},
    )()
    model = stacked_AttRec(
        args,
        stack_num=stack_num,
        head_num=head_num,
        time_info_mode=time_info_mode,
        window_size=window_size,
        is_pretrained_item_weight=False,
        bpr_item_weight=None,
        isItemGrad=True,
        isUserBN=False,
    )
    model.eval()
    return model


def _close(left, right, atol=1e-5, rtol=1e-5):
    return torch.allclose(left, right, atol=atol, rtol=rtol)


def _gap(left, right):
    return (left - right).abs().max().item()


def test_ordinary_batch_normalizes_over_history():
    latent = 4
    with legacy_cuda_stays_on_cpu():
        model = make_model(4, 8, latent, stack_num=0, head_num=1, time_info_mode=0, window_size=3)
    items = torch.zeros(8, latent)
    items[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    items[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    items[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    items[3] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    items[4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    users_w = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0],
            [10.0, 1.0, 0.5, 0.0],
            [0.2, 4.0, 1.0, 0.0],
            [3.0, 0.3, 2.0, 0.0],
        ]
    )
    model.item_embedding.weight.data.copy_(items)
    model.user_embedding.weight.data.copy_(users_w)
    users = torch.arange(4)
    hist = torch.tensor([[0, 1, 2]]).expand(4, 3).contiguous()
    pos = torch.full((4,), 3)
    neg = torch.full((4,), 4)
    with torch.no_grad():
        batched, _ = model(users, hist, pos, neg)
    expected = torch.ones(4)
    assert _close(batched, expected), (
        f"history weights do not sum to 1 per user: {batched.tolist()}"
    )


def test_ordinary_chunk_invariance_without_user_batchnorm():
    torch.manual_seed(0)
    num_users, num_items, latent, window = 4, 10, 8, 5
    with legacy_cuda_stays_on_cpu():
        model = make_model(
            num_users, num_items, latent, stack_num=2, head_num=2, time_info_mode=0, window_size=window
        )
    model.user_embedding.weight.data.copy_(torch.randn(num_users, latent))
    model.item_embedding.weight.data.copy_(torch.randn(num_items, latent))
    users = torch.arange(num_users)
    hist = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [1, 3, 5, 7, 9],
            [0, 2, 4, 6, 8],
        ]
    )
    pos = torch.tensor([1, 2, 3, 4])
    neg = torch.tensor([5, 6, 7, 8])
    with torch.no_grad():
        full_pos, full_neg = model(users, hist, pos, neg)
        chunk_pos, chunk_neg = [], []
        for start in (0, 2):
            sl = slice(start, start + 2)
            p, n = model(users[sl], hist[sl], pos[sl], neg[sl])
            chunk_pos.append(p)
            chunk_neg.append(n)
        one_pos, one_neg = [], []
        for i in range(num_users):
            sl = slice(i, i + 1)
            p, n = model(users[sl], hist[sl], pos[sl], neg[sl])
            one_pos.append(p)
            one_neg.append(n)
        dup_pos, dup_neg = model(
            torch.cat([users, users]),
            torch.cat([hist, hist]),
            torch.cat([pos, pos]),
            torch.cat([neg, neg]),
        )
    errors = []
    cat_pos, cat_neg = torch.cat(chunk_pos), torch.cat(chunk_neg)
    one_pos_t, one_neg_t = torch.cat(one_pos), torch.cat(one_neg)
    if not _close(full_pos, cat_pos) or not _close(full_neg, cat_neg):
        errors.append(
            f"chunk2 gap pos={_gap(full_pos, cat_pos)} neg={_gap(full_neg, cat_neg)}"
        )
    if not _close(full_pos, one_pos_t) or not _close(full_neg, one_neg_t):
        errors.append(
            f"chunk1 gap pos={_gap(full_pos, one_pos_t)} neg={_gap(full_neg, one_neg_t)}"
        )
    if not _close(dup_pos[:num_users], full_pos) or not _close(dup_neg[:num_users], full_neg):
        errors.append(
            f"duplication gap pos={_gap(dup_pos[:num_users], full_pos)} neg={_gap(dup_neg[:num_users], full_neg)}"
        )
    assert not errors, "; ".join(errors)


def test_time5_chunk_and_duplication_invariance_without_user_batchnorm():
    torch.manual_seed(1)
    num_users, num_items, latent, window = 4, 12, 8, 5
    with legacy_cuda_stays_on_cpu():
        model = make_model(
            num_users, num_items, latent, stack_num=1, head_num=2, time_info_mode=5, window_size=window
        )
        assert model.position_enc.device.type == "cpu"
        model.user_embedding.weight.data.copy_(torch.randn(num_users, latent))
        model.item_embedding.weight.data.copy_(torch.randn(num_items, latent))
        users = torch.arange(num_users)
        hist = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [2, 4, 6, 8, 10],
                [1, 3, 5, 7, 11],
            ]
        )
        pos = torch.tensor([1, 2, 3, 4])
        neg = torch.tensor([6, 7, 8, 9])
        with torch.no_grad():
            full_pos, full_neg = model(users, hist, pos, neg)
            one_pos, one_neg = [], []
            for i in range(num_users):
                sl = slice(i, i + 1)
                p, n = model(users[sl], hist[sl], pos[sl], neg[sl])
                one_pos.append(p)
                one_neg.append(n)
            pair_pos, pair_neg = [], []
            for start in (0, 2):
                sl = slice(start, start + 2)
                p, n = model(users[sl], hist[sl], pos[sl], neg[sl])
                pair_pos.append(p)
                pair_neg.append(n)
            dup_pos, dup_neg = model(
                torch.cat([users, users]),
                torch.cat([hist, hist]),
                torch.cat([pos, pos]),
                torch.cat([neg, neg]),
            )
    errors = []
    one_pos_t, one_neg_t = torch.cat(one_pos), torch.cat(one_neg)
    pair_pos_t, pair_neg_t = torch.cat(pair_pos), torch.cat(pair_neg)
    if not _close(full_pos, one_pos_t) or not _close(full_neg, one_neg_t):
        errors.append(
            f"chunk1 gap pos={_gap(full_pos, one_pos_t)} neg={_gap(full_neg, one_neg_t)}"
        )
    if not _close(full_pos, pair_pos_t) or not _close(full_neg, pair_neg_t):
        errors.append(
            f"chunk2 gap pos={_gap(full_pos, pair_pos_t)} neg={_gap(full_neg, pair_neg_t)}"
        )
    if not _close(dup_pos[:num_users], full_pos) or not _close(dup_pos[num_users:], full_pos):
        errors.append(f"duplication gap pos={_gap(dup_pos[:num_users], full_pos)}")
    if not _close(dup_neg[:num_users], full_neg):
        errors.append(f"duplication gap neg={_gap(dup_neg[:num_users], full_neg)}")
    assert not errors, "; ".join(errors)


def _loss_value_node(source):
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "loss":
                found.append(node.value)
    if len(found) != 1:
        raise AssertionError(f"expected one loss assignment, found {len(found)}")
    return found[0]


def _eval_loss(expr, pos_pref_score, neg_pref_score):
    expression = ast.fix_missing_locations(ast.Expression(expr))
    code = compile(expression, "main.py", "eval")
    return eval(
        code,
        {"__builtins__": {}},
        {
            "F": torch.nn.functional,
            "pos_pref_score": pos_pref_score,
            "neg_pref_score": neg_pref_score,
        },
    )


def test_extracted_main_loss_matches_analytic_value_gradient_and_extremes():
    assert "main" not in sys.modules
    source = (ROOT / "main.py").read_bytes().decode("utf-8")
    expr = _loss_value_node(source)
    pos = torch.tensor(
        [0.0, 2.0, -3.0, 40.0, -80.0, 1000.0, -1000.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    neg = torch.tensor(
        [0.0, -1.0, 1.0, -50.0, 40.0, -1000.0, 1000.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = _eval_loss(expr, pos, neg)
    diff = pos.detach() - neg.detach()
    analytic = torch.nn.functional.softplus(-diff).mean()
    naive = -torch.log(torch.sigmoid(diff))
    errors = []
    if torch.isfinite(naive).all():
        errors.append("fixture expected naive -log(sigmoid) overflow at an extreme score")
    if not torch.isfinite(loss):
        errors.append(f"extracted loss is non-finite: {loss}")
    if not _close(loss, analytic, atol=1e-6, rtol=1e-6):
        errors.append(f"value loss={float(loss)} analytic={float(analytic)}")
    loss.backward()
    grad = (torch.sigmoid(diff) - 1.0) / diff.numel()
    if not _close(pos.grad, grad, atol=1e-6, rtol=1e-5):
        errors.append(f"pos grad max gap={_gap(pos.grad, grad)}")
    if not _close(neg.grad, -grad, atol=1e-6, rtol=1e-5):
        errors.append(f"neg grad max gap={_gap(neg.grad, -grad)}")
    assert not errors, "; ".join(errors)


def _is_neg_logsigmoid_mean(expr):
    # The required spelling is -F.logsigmoid(pos_pref_score - neg_pref_score).mean().
    # Call binding is tighter than unary minus, so the parse is -(logsigmoid(...).mean()).
    # That equals mean(-logsigmoid(...)) because mean is linear; the numeric test checks it.
    if not isinstance(expr, ast.UnaryOp) or not isinstance(expr.op, ast.USub):
        return False
    mean_call = expr.operand
    if not isinstance(mean_call, ast.Call) or mean_call.args or mean_call.keywords:
        return False
    if not isinstance(mean_call.func, ast.Attribute) or mean_call.func.attr != "mean":
        return False
    log_call = mean_call.func.value
    if not isinstance(log_call, ast.Call) or log_call.keywords or len(log_call.args) != 1:
        return False
    func = log_call.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "logsigmoid"
        and isinstance(func.value, ast.Name)
        and func.value.id == "F"
    ):
        return False
    arg = log_call.args[0]
    return (
        isinstance(arg, ast.BinOp)
        and isinstance(arg.op, ast.Sub)
        and isinstance(arg.left, ast.Name)
        and arg.left.id == "pos_pref_score"
        and isinstance(arg.right, ast.Name)
        and arg.right.id == "neg_pref_score"
    )


def test_main_loss_ast_is_neg_logsigmoid_mean():
    assert "main" not in sys.modules
    source = (ROOT / "main.py").read_bytes().decode("utf-8")
    expr = _loss_value_node(source)
    assert _is_neg_logsigmoid_mean(expr), ast.dump(expr)
