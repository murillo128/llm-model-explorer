"""Adapt a narrow native Codex profile to the shared App Server protocol.

Standalone profiles are read from the server's reported CODEX_HOME. Only model,
reasoning effort and the auxiliary concurrency cap cross this adapter; permission
policy remains owned by the existing Skillforge launcher.
"""

from pathlib import Path
import re
import tomllib


EFFORTS = {name: index for index, name in enumerate(
    ("low", "medium", "high", "xhigh", "max", "ultra")
)}
CAP_KEY = "agents.max_concurrent_threads_per_session"


class ProfilePolicy:
    def __init__(self, client, codex_home, name, role):
        if role not in {"execution", "audit"}:
            raise ValueError("Unknown Codex launcher role")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Codex profile must be a basename with letters, digits, '_' or '-'")
        if not isinstance(codex_home, str) or not Path(codex_home).is_absolute():
            raise ValueError("App Server did not report an absolute CODEX_HOME")
        self.source = Path(codex_home) / f"{name}.config.toml"
        self.name = name
        self.role = role
        try:
            with self.source.open("rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot load Codex profile {self.source}: {type(exc).__name__}") from None
        agents = config.get("agents")
        if (set(config) != {"model", "model_reasoning_effort", "agents"}
                or not isinstance(agents, dict)
                or set(agents) != {"max_concurrent_threads_per_session"}
                or type(agents["max_concurrent_threads_per_session"]) is not int
                or agents["max_concurrent_threads_per_session"] != 3):
            raise ValueError("Codex profile permits only model, model_reasoning_effort and agents.max_concurrent_threads_per_session = 3")
        self.model = config["model"]
        self.effort = config["model_reasoning_effort"]
        floor = "max" if role == "audit" else "xhigh"
        if name == "codex-critical":
            floor = "ultra"
        if (self.model != "gpt-6-astra" or not isinstance(self.effort, str)
                or self.effort not in EFFORTS or EFFORTS[self.effort] < EFFORTS[floor]):
            raise ValueError(f"The {role} profile requires gpt-6-astra with at least {floor} reasoning")
        self.available_efforts = set()
        cursor = None
        seen = set()
        while True:
            page = client.request("model/list", {"includeHidden": True, "cursor": cursor})
            for model in page.get("data", []):
                if model.get("model") == self.model:
                    self.available_efforts.update(
                        item.get("reasoningEffort") for item in model.get("supportedReasoningEfforts", [])
                    )
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if not isinstance(cursor, str) or cursor in seen:
                raise ValueError("model/list returned an invalid pagination cursor")
            seen.add(cursor)
        if self.effort not in self.available_efforts:
            raise ValueError(f"Requested capability unavailable: {self.model} / {self.effort}; no fallback selected")

    def thread_options(self, *, resumed=False):
        config = {CAP_KEY: 3}
        if resumed:
            return {"config": config}
        config["model_reasoning_effort"] = self.effort
        return {"model": self.model, "config": config, "allowProviderModelFallback": False}

    def confirm(self, response, *, resumed=False):
        model = response.get("model")
        effort = response.get("reasoningEffort")
        acceptable = (model == self.model and isinstance(effort, str)
                      and effort in self.available_efforts
                      and effort in EFFORTS and EFFORTS[effort] >= EFFORTS[self.effort])
        if not resumed:
            acceptable = acceptable and effort == self.effort
        if not acceptable:
            instruction = (
                " Explicitly adopt the desired model/effort on this inactive thread before retrying; "
                "the launcher will not replace its saved selection."
                if resumed else " No turn was started."
            )
            raise ValueError("App Server model/effort does not satisfy the requested profile." + instruction)
        self.confirmed_model = model
        self.confirmed_effort = effort

    def turn_options(self):
        return {"model": self.confirmed_model, "effort": self.confirmed_effort}

    def log_confirmation(self, log, *, thread_id, cwd, resumed):
        log("model_policy_confirmed", role=self.role, profile=self.name,
            source=str(self.source), requestedModel=self.model, requestedEffort=self.effort,
            confirmedModel=self.confirmed_model, confirmedEffort=self.confirmed_effort,
            requestedMaxAuxiliaryThreads=3, threadId=thread_id, cwd=cwd, resumed=resumed)
