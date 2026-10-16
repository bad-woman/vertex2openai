import asyncio
import os
import sys
import time
import unittest
from unittest.mock import patch

from starlette.requests import Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import resumable
from routes.chat_api import _resume_start_seq


class ResumableTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_unkeyed_identical_requests_do_not_share_a_buffer(self):
        first, _ = resumable.compute_stream_id({}, "api-key", "same-body")
        second, _ = resumable.compute_stream_id({}, "api-key", "same-body")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("rsm-"))

    def test_explicit_idempotency_key_is_stable(self):
        headers = {"x-idempotency-key": "retry-42"}
        first, _ = resumable.compute_stream_id(headers, "api-key", "same-body")
        second, _ = resumable.compute_stream_id(headers, "api-key", "same-body")
        self.assertEqual(first, second)

    def test_response_resume_id_can_be_sent_back(self):
        stream_id = "rsm-" + "a" * 24
        actual, source = resumable.compute_stream_id({"x-resume-id": stream_id}, "api-key", "same-body")
        self.assertEqual(actual, stream_id)
        self.assertIn("X-Resume-Id", source)

    def test_last_event_id_resumes_after_last_delivered_event(self):
        request = Request({
            "type": "http", "method": "POST", "path": "/v1/chat/completions",
            "headers": [(b"last-event-id", b"7")], "query_string": b"",
        })
        self.assertEqual(_resume_start_seq(request), 8)

    async def test_resume_id_rejects_different_owner_or_body(self):
        store = resumable.ResumableStore()
        await store.create_if_absent("rsm-" + "b" * 24, owner_hash="owner-a", fingerprint_hash="body-a")
        with self.assertRaises(resumable.ResumeIdentityMismatch):
            await store.create_if_absent("rsm-" + "b" * 24, owner_hash="owner-b", fingerprint_hash="body-a")
        with self.assertRaises(resumable.ResumeIdentityMismatch):
            await store.create_if_absent("rsm-" + "b" * 24, owner_hash="owner-a", fingerprint_hash="body-b")

    async def test_does_not_evict_a_live_buffer_when_count_limit_is_reached(self):
        store = resumable.ResumableStore()
        with patch.object(resumable, "max_streams", return_value=1):
            await store.create_if_absent("rsm-" + "c" * 24)
            with self.assertRaises(resumable.ResumableCapacityError):
                await store.create_if_absent("rsm-" + "d" * 24)

    async def test_stream_byte_limit_is_enforced(self):
        store = resumable.ResumableStore()
        buf, _ = await store.create_if_absent("rsm-" + "e" * 24)
        with patch.object(resumable, "max_stream_bytes", return_value=4):
            with self.assertRaises(resumable.ResumableCapacityError):
                await store.append(buf, "12345")

    async def test_sweeper_cancels_expired_live_pump(self):
        store = resumable.ResumableStore()
        buf, _ = await store.create_if_absent("rsm-" + "f" * 24)
        task = asyncio.create_task(asyncio.sleep(60))
        store.register_pump(buf.id, task)
        buf.created_at = time.time() - 7201
        await store.sweep()
        await asyncio.sleep(0)
        self.assertTrue(task.cancelled())


if __name__ == "__main__":
    unittest.main()
