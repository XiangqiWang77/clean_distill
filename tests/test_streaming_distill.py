"""Regressions for memory-bounded persistent distillation."""

from __future__ import annotations

import unittest

import torch

from src.clean_self_distill.streaming_distill import stream_distillation_chunks
from src.clean_self_distill.train_eval import _same_prefix_distillation_terms


def _realized_logprob_sum(logits: torch.Tensor, labels: torch.Tensor) -> float:
    selected = torch.log_softmax(logits.detach().float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    )
    return float(selected.sum().item())


class StreamingDistillationTests(unittest.TestCase):
    def _assert_matches_unchunked(
        self, *, top_k: int, temperature: float, token_clip: float
    ) -> None:
        torch.manual_seed(731)
        dtype = torch.float64
        token_count, input_size, hidden_size, vocab_size = 7, 4, 5, 6
        inputs = torch.randn(1, token_count, input_size, dtype=dtype)
        labels = torch.tensor([[0, 5, 1, 3, 2, 4, 0]])
        teacher_hidden_map = torch.randn(
            hidden_size, vocab_size, dtype=dtype
        )
        teacher_bias = torch.randn(vocab_size, dtype=dtype)

        reference_backbone = torch.nn.Linear(
            input_size, hidden_size, bias=True, dtype=dtype
        )
        reference_head = torch.nn.Linear(
            hidden_size, vocab_size, bias=True, dtype=dtype
        )
        streamed_backbone = torch.nn.Linear(
            input_size, hidden_size, bias=True, dtype=dtype
        )
        streamed_head = torch.nn.Linear(
            hidden_size, vocab_size, bias=True, dtype=dtype
        )
        streamed_backbone.load_state_dict(reference_backbone.state_dict())
        streamed_head.load_state_dict(reference_head.state_dict())
        reference_head.requires_grad_(False)
        streamed_head.requires_grad_(False)

        reference_hidden = reference_backbone(inputs)
        reference_student = reference_head(reference_hidden)
        reference_teacher = (
            reference_student.detach() * 0.17
            + reference_hidden.detach() @ teacher_hidden_map
            + teacher_bias
        )
        reference_loss, reference_kl = _same_prefix_distillation_terms(
            reference_student,
            reference_teacher,
            top_k=top_k,
            temperature=temperature,
            token_clip=token_clip,
        )
        reference_loss.backward()

        streamed_hidden = streamed_backbone(inputs)

        def teacher_for_chunk(student, hidden, _start, _stop):
            return student * 0.17 + hidden @ teacher_hidden_map + teacher_bias

        result = stream_distillation_chunks(
            streamed_hidden,
            labels,
            streamed_head,
            teacher_for_chunk,
            chunk_size=3,
            top_k=top_k,
            temperature=temperature,
            token_clip=token_clip,
            backward=True,
        )

        self.assertEqual(result.token_count, token_count)
        self.assertEqual(result.max_chunk_tokens, 3)
        # Chunked reductions change float32 summation order.  Require normal
        # float32 agreement rather than an unattainable 1e-11 absolute match.
        self.assertAlmostEqual(result.loss, float(reference_loss.item()), places=6)
        self.assertAlmostEqual(
            result.mean_kl, float(reference_kl.mean().item()), places=6
        )
        reference_student_logprob = _realized_logprob_sum(
            reference_student, labels
        )
        reference_teacher_logprob = _realized_logprob_sum(
            reference_teacher, labels
        )
        self.assertAlmostEqual(
            result.student_logprob_sum, reference_student_logprob, places=5
        )
        self.assertAlmostEqual(
            result.teacher_logprob_sum, reference_teacher_logprob, places=5
        )
        self.assertAlmostEqual(
            result.student_normalized_logprob,
            reference_student_logprob / token_count,
            places=6,
        )
        self.assertAlmostEqual(
            result.teacher_normalized_logprob,
            reference_teacher_logprob / token_count,
            places=6,
        )

        for reference_parameter, streamed_parameter in zip(
            reference_backbone.parameters(), streamed_backbone.parameters()
        ):
            torch.testing.assert_close(
                streamed_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-7,
            )

    def test_uneven_chunks_match_full_objective_and_gradients(self):
        for top_k in (1, 2, 6):
            for temperature in (0.7, 1.0, 1.9):
                for token_clip in (0.0, 0.08):
                    with self.subTest(
                        top_k=top_k,
                        temperature=temperature,
                        token_clip=token_clip,
                    ):
                        self._assert_matches_unchunked(
                            top_k=top_k,
                            temperature=temperature,
                            token_clip=token_clip,
                        )

    def test_observer_receives_exact_detached_chunks_after_backward(self):
        torch.manual_seed(19)
        hidden = torch.randn(1, 5, 3, requires_grad=True)
        labels = torch.tensor([[1, 0, 3, 2, 1]])
        head = torch.nn.Linear(3, 4, bias=False)
        head.requires_grad_(False)
        teacher_offset = torch.tensor([0.1, -0.4, 0.7, 0.2])
        observations = []
        upstream_backward_calls = []
        hidden.register_hook(lambda gradient: upstream_backward_calls.append(gradient))

        def teacher_for_chunk(student, _hidden, start, stop):
            return student * 0.25 + teacher_offset + (start + stop) * 0.01

        def observer(start, stop, student, teacher, per_token_kl, chunk_labels):
            # Chunk-local autograd must not traverse the shared upstream graph.
            self.assertEqual(upstream_backward_calls, [])
            for tensor in (student, teacher, per_token_kl, chunk_labels):
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)
            observations.append(
                (
                    start,
                    stop,
                    student.clone(),
                    teacher.clone(),
                    per_token_kl.clone(),
                    chunk_labels.clone(),
                )
            )

        initial_weight = head.weight.detach().clone()
        result = stream_distillation_chunks(
            hidden,
            labels,
            head,
            teacher_for_chunk,
            chunk_size=2,
            top_k=2,
            temperature=1.3,
            token_clip=0.0,
            backward=True,
            observer=observer,
        )
        full_student = hidden.detach() @ initial_weight.t()
        expected_teacher_chunks = []
        for start in range(0, 5, 2):
            stop = min(start + 2, 5)
            expected_teacher_chunks.append(
                full_student[:, start:stop] * 0.25
                + teacher_offset
                + (start + stop) * 0.01
            )
        full_teacher = torch.cat(expected_teacher_chunks, dim=1)
        _, expected_kl = _same_prefix_distillation_terms(
            full_student,
            full_teacher,
            top_k=2,
            temperature=1.3,
            token_clip=0.0,
        )

        self.assertEqual([(row[0], row[1]) for row in observations], [(0, 2), (2, 4), (4, 5)])
        torch.testing.assert_close(
            torch.cat([row[2] for row in observations], dim=1), full_student
        )
        torch.testing.assert_close(
            torch.cat([row[3] for row in observations], dim=1), full_teacher
        )
        torch.testing.assert_close(
            torch.cat([row[4] for row in observations], dim=1), expected_kl
        )
        torch.testing.assert_close(
            torch.cat([row[5] for row in observations], dim=1), labels
        )
        self.assertEqual(result.max_chunk_tokens, 2)
        self.assertEqual(len(upstream_backward_calls), 1)
        self.assertIsNotNone(hidden.grad)

    def test_projection_never_exceeds_chunk_size(self):
        hidden = torch.randn(1, 11, 3, requires_grad=True)
        labels = torch.arange(11).remainder(5).unsqueeze(0)
        projected_lengths = []
        teacher_lengths = []
        projection = torch.nn.Linear(3, 5)

        def project(chunk):
            projected_lengths.append(int(chunk.shape[1]))
            return projection(chunk)

        def teacher(student, hidden_chunk, _start, _stop):
            teacher_lengths.append(
                (int(student.shape[1]), int(hidden_chunk.shape[1]))
            )
            return student + 0.2

        result = stream_distillation_chunks(
            hidden,
            labels,
            project,
            teacher,
            chunk_size=4,
            top_k=5,
            temperature=1.0,
            token_clip=0.0,
            backward=False,
        )

        self.assertEqual(projected_lengths, [4, 4, 3])
        self.assertEqual(teacher_lengths, [(4, 4), (4, 4), (3, 3)])
        self.assertEqual(result.max_chunk_tokens, 4)

    def test_rejects_mismatched_labels_and_teacher_shape(self):
        hidden = torch.randn(1, 3, 2)
        labels = torch.zeros(1, 2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "labels"):
            stream_distillation_chunks(
                hidden,
                labels,
                lambda chunk: torch.zeros(1, chunk.shape[1], 3),
                lambda student, _hidden, _start, _stop: student,
                2,
                2,
                1.0,
                0.0,
                False,
            )

        labels = torch.zeros(1, 3, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "exactly"):
            stream_distillation_chunks(
                hidden,
                labels,
                lambda chunk: torch.zeros(1, chunk.shape[1], 3),
                lambda student, _hidden, _start, _stop: student[..., :2],
                2,
                2,
                1.0,
                0.0,
                False,
            )


if __name__ == "__main__":
    unittest.main()
