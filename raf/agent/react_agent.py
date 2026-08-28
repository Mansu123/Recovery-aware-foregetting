"""ReAct agent over an AppWorld `world`, with RA-FM long-term memory attached."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import prompts
from ..llm.base import LLM
from ..memory.store import MemoryStore

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _extract_code(text: str) -> str:
    m = _CODE.search(text)
    if m:
        return m.group(1).strip()
    # model forgot the fence: take everything that looks like code
    return text.strip().strip("`").strip()


@dataclass
class TaskResult:
    task_id: str
    instruction: str
    completed: bool
    num_steps: int
    answer: str | None = None
    transcript: list[dict] = field(default_factory=list)
    eval: dict | None = None
    memories_written: int = 0
    memories_recalled: int = 0


class ReactAgent:
    def __init__(self, llm: LLM, memory: MemoryStore, *,
                 max_interactions: int = 40, max_history_chars: int = 8000):
        self.llm = llm
        self.memory = memory
        self.max_interactions = max_interactions
        self.max_history_chars = max_history_chars

    # ------------------------------------------------------------------ #
    def run(self, world) -> TaskResult:
        task = world.task
        sup = task.supervisor
        instr = task.instruction

        recalled = self.memory.retrieve(instr, k=self.memory.cfg.retrieve_k)
        mem_txt = ("\n".join(f"- [{m.kind}] {m.text}" for m in recalled)
                   if recalled else "(none)")

        header = prompts.TASK_HEADER.format(
            first=sup.first_name, last=sup.last_name, email=sup.email,
            instruction=instr)
        header += prompts.MEMORY_BLOCK.format(memories=mem_txt)

        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": header +
             "\n\nBegin. Send your first python code block."},
        ]
        transcript: list[dict] = []
        output: str | None = None
        completed = False

        for step in range(self.max_interactions):
            reply = self.llm.chat(messages, stop=["```\n\n", "\nObservation:"])
            code = _extract_code(reply)
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})

            output = world.execute(code)
            transcript.append({"step": step, "code": code, "output": output})

            # write salient observations to long-term memory
            self._observe(code, output, task.id, instr)

            obs = output if len(output) < 4000 else output[:2000] + "\n...\n" + output[-1500:]
            messages.append({"role": "user", "content": f"Output:\n{obs}"})
            self._trim(messages)

            if world.task_completed():
                completed = True
                break

        answer = self._final_answer(transcript)
        written = self._reflect(messages, task.id, instr)

        result = TaskResult(
            task_id=task.id, instruction=instr, completed=completed,
            num_steps=len(transcript), answer=answer, transcript=transcript,
            memories_written=written, memories_recalled=len(recalled))
        try:
            result.eval = world.evaluate().to_dict()
        except Exception:  # noqa: BLE001
            result.eval = None
        return result

    # ------------------------------------------------------------------ #
    def _observe(self, code: str, output: str, task_id: str, instr: str) -> None:
        out = output.strip()
        if not out or out.lower().startswith("execution failed"):
            # still worth remembering a hard error once
            if "Traceback" in out:
                self.memory.add(f"While `{code.splitlines()[0][:80]}` failed: "
                                f"{out.splitlines()[-1][:160]}",
                                kind="outcome", task_id=task_id)
            return
        snippet = out if len(out) < 400 else out[:400]
        # heuristic: lines that carry ids / values / credentials are facts
        kind = "fact" if re.search(
            r"(id|email|password|token|phone|address|amount|price|balance)",
            snippet, re.I) else "observation"
        self.memory.add(
            f"For sub-goal via `{code.splitlines()[0][:70]}` -> {snippet}",
            kind=kind, task_id=task_id,
            meta={"goal_relevant": any(w in instr.lower()
                                      for w in snippet.lower().split()[:5])})

    # ------------------------------------------------------------------ #
    def _reflect(self, messages, task_id: str, instr: str) -> int:
        msgs = messages + [{"role": "user", "content": prompts.REFLECT}]
        try:
            notes = self.llm.chat(msgs, max_new_tokens=256, temperature=0.0)
        except Exception:  # noqa: BLE001
            return 0
        n = 0
        for line in notes.splitlines():
            line = line.strip("-*• \t")
            if len(line) < 8:
                continue
            self.memory.add(line, kind="fact", task_id=task_id,
                            meta={"goal_relevant": True, "source": "reflection"})
            n += 1
        return n

    # ------------------------------------------------------------------ #
    @staticmethod
    def _final_answer(transcript: list[dict]) -> str | None:
        pat = re.compile(r"complete_task\([^)]*answer\s*=\s*(.+?)\s*[,)]", re.S)
        for turn in reversed(transcript):
            m = pat.search(turn["code"])
            if m:
                return m.group(1).strip().strip("'\"")
        return None

    # ------------------------------------------------------------------ #
    def _trim(self, messages: list[dict]) -> None:
        # keep system + first user + a sliding window of recent turns
        while (sum(len(m["content"]) for m in messages) > self.max_history_chars
               and len(messages) > 5):
            del messages[2:4]
