use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tracing::warn;

use crate::config_validation::sanitize_role_name;
use crate::event::{AgentEvent, AgentType, DelegateSignal, EventType, WorkDoneSignal};

const MAX_RECENT_EVENTS: usize = 50;
const MAX_FIRST_PROMPTS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    Thinking,
    Working,
    Compacting,
    WaitingForInput,
    Idle,
    Error,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DashboardStats {
    pub active: usize,
    pub working: usize,
    pub thinking: usize,
    pub waiting: usize,
    pub errors: usize,
    pub idle: usize,
    pub compacting: usize,
    pub total_tools: u64,
}

#[derive(Debug, Clone)]
pub struct ActiveTool {
    pub name: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub session_id: String,
    pub agent_type: AgentType,
    pub cwd: Option<String>,
    pub status: SessionStatus,
    pub active_tool: Option<ActiveTool>,
    pub started_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub recent_events: VecDeque<AgentEvent>,
    pub tool_count: u32,
    pub last_user_prompt: Option<String>,
    pub first_prompts: Vec<String>,
    pub pane_id: Option<String>,
    /// Git branch name of the pane's cwd, or dir basename if not a git repo.
    /// None until first refresh.
    pub git_branch: Option<String>,
    /// Last time the branch was refreshed.
    pub git_branch_refreshed_at: Option<DateTime<Utc>>,
    /// Linked git worktree name, or None when on the main checkout / not a worktree.
    pub git_worktree: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct AppState {
    pub sessions: HashMap<String, SessionState>,
    /// Remembers started_at per pane so a `/clear` restart keeps its position.
    pane_started_at: HashMap<String, DateTime<Utc>>,
    /// Set by the background version-check task when a newer release exists.
    pub update_available: Option<String>,
    /// Pane IDs created by our app — events from unknown panes are rejected.
    pub managed_pane_ids: HashSet<String>,
    /// Maps pane_id → orchestration role name (set when orchestration tab opens).
    pub pane_role_map: HashMap<String, String>,
    /// Maps pane_id → working directory for orchestration panes.
    pub pane_cwd_map: HashMap<String, String>,
    /// Pane IDs that are orchestrator (start=true) roles — only these can delegate.
    pub orchestrator_pane_ids: HashSet<String>,
    /// Delegate signals from the orchestrator, consumed by dispatch (M5).
    pub delegate_events: Vec<DelegateSignal>,
    /// Work-done signals from workers (or orchestrator --done), consumed by feedback (M5b).
    pub work_done_events: Vec<WorkDoneSignal>,
}

pub type SharedState = Arc<RwLock<AppState>>;

impl AppState {
    fn is_interactive_tool(name: &str) -> bool {
        matches!(name, "AskUserQuestion" | "ExitPlanMode")
    }

    pub fn aggregate_stats(&self) -> DashboardStats {
        let mut stats = DashboardStats::default();
        for session in self.sessions.values() {
            if session.agent_type == AgentType::None {
                continue;
            }
            stats.active += 1;
            match session.status {
                SessionStatus::Working => stats.working += 1,
                SessionStatus::Thinking => stats.thinking += 1,
                SessionStatus::WaitingForInput => stats.waiting += 1,
                SessionStatus::Error => stats.errors += 1,
                SessionStatus::Idle => stats.idle += 1,
                SessionStatus::Compacting => stats.compacting += 1,
            }
            stats.total_tools += session.tool_count as u64;
        }
        stats
    }

    /// Register a pane ID as managed by our app.
    pub fn register_pane(&mut self, pane_id: String) {
        self.managed_pane_ids.insert(pane_id);
    }

    /// Create a placeholder session for a newly created pane so it always has a dashboard card.
    pub fn insert_placeholder_session(&mut self, pane_id: String, cwd: Option<String>) {
        let session_id = format!("pane-{}", pane_id);
        let now = Utc::now();
        let started_at = self.pane_started_at.get(&pane_id).copied().unwrap_or(now);
        self.sessions.insert(
            session_id.clone(),
            SessionState {
                session_id,
                agent_type: AgentType::None,
                cwd,
                status: SessionStatus::Idle,
                active_tool: None,
                started_at,
                last_activity: now,
                recent_events: VecDeque::new(),
                tool_count: 0,
                last_user_prompt: None,
                first_prompts: Vec::new(),
                pane_id: Some(pane_id),
                git_branch: None,
                git_branch_refreshed_at: None,
                git_worktree: None,
            },
        );
    }

    /// Unregister a pane ID (e.g., when closing a pane).
    pub fn unregister_pane(&mut self, pane_id: &str) {
        self.managed_pane_ids.remove(pane_id);
        self.pane_role_map.remove(pane_id);
        self.pane_cwd_map.remove(pane_id);
        self.orchestrator_pane_ids.remove(pane_id);
    }

    /// Handle a delegate signal from the orchestrator.
    /// Validates that the sender is an orchestrator (start=true) role before enqueuing.
    pub fn handle_delegate(&mut self, signal: DelegateSignal) {
        if !self.pane_role_map.contains_key(&signal.pane_id) {
            warn!(pane_id = %signal.pane_id, "delegate from unknown pane");
            return;
        }
        if !self.orchestrator_pane_ids.contains(&signal.pane_id) {
            let role = self
                .pane_role_map
                .get(&signal.pane_id)
                .cloned()
                .unwrap_or_default();
            warn!(pane_id = %signal.pane_id, role = %role, "delegate from non-orchestrator pane");
            return;
        }
        self.delegate_events.push(signal);
    }

    /// Handle a work-done signal from a worker (or orchestrator --done).
    /// Resolves pane_id → role name, writes a per-role summary file, and
    /// stores the signal for feedback to the orchestrator (M5b).
    pub fn handle_work_done(&mut self, signal: WorkDoneSignal) {
        let role_name = match self.pane_role_map.get(&signal.pane_id) {
            Some(name) => name.clone(),
            None => {
                warn!(pane_id = %signal.pane_id, "work-done from unknown pane");
                return;
            }
        };

        // Write summary to .dot-agent-deck/work-done-{role}.md
        if let Some(cwd) = self.pane_cwd_map.get(&signal.pane_id) {
            let safe_name = sanitize_role_name(&role_name);
            let dir = std::path::Path::new(cwd).join(".dot-agent-deck");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                warn!(dir = %dir.display(), role = %role_name, error = %e, "failed to create work-done directory");
            }
            let file_path = dir.join(format!("work-done-{safe_name}.md"));
            if let Err(e) = std::fs::write(&file_path, &signal.task) {
                warn!(path = %file_path.display(), role = %role_name, error = %e, "failed to write work-done summary");
            }
        }

        self.work_done_events.push(signal);
    }

    /// Refresh git_branch for all sessions whose branch is stale (older than 30s).
    // Note: git subprocess runs synchronously inside the write lock. This is acceptable
    // for local repos (typically <10ms) but may cause brief UI stutter on slow network
    // mounts. Non-blocking async refresh is a future improvement.
    pub fn refresh_stale_branches(&mut self) {
        let now = Utc::now();
        let threshold = chrono::Duration::seconds(30);
        for session in self.sessions.values_mut() {
            let needs_refresh = match session.git_branch_refreshed_at {
                None => true,
                Some(t) => (now - t) > threshold,
            };
            if needs_refresh {
                refresh_git_branch(session);
            }
        }
    }

    pub fn apply_event(&mut self, mut event: AgentEvent) {
        // Only accept events from panes managed by our app.
        // Events without a pane_id (external agents) are rejected when we have
        // managed panes. Events with an unknown pane_id are rejected unless it
        // is a SessionStart (which may arrive before register_pane during startup).
        if let Some(ref pane_id) = event.pane_id {
            if !self.managed_pane_ids.contains(pane_id) {
                if event.event_type == EventType::SessionStart {
                    // Auto-register the pane to handle the startup race where
                    // the hook fires before register_pane is called.
                    self.managed_pane_ids.insert(pane_id.clone());
                } else {
                    return;
                }
            }
        } else if !self.managed_pane_ids.is_empty() {
            return;
        }
        if let Some(ref pane_id) = event.pane_id
            && let Some(existing_id) = self.sessions.iter().find_map(|(id, session)| {
                (session.pane_id.as_ref().is_some_and(|p| p == pane_id) && id != &event.session_id)
                    .then(|| id.clone())
            })
        {
            let old_id = std::mem::replace(&mut event.session_id, existing_id);
            if old_id != event.session_id {
                self.sessions.remove(&old_id);
            }
        }

        if event.event_type == EventType::SessionEnd {
            // Preserve started_at for the pane so a restarted session keeps its position.
            let pane_id_and_cwd = self.sessions.get(&event.session_id).and_then(|session| {
                session.pane_id.as_ref().map(|pid| {
                    self.pane_started_at.insert(pid.clone(), session.started_at);
                    (pid.clone(), session.cwd.clone())
                })
            });
            self.sessions.remove(&event.session_id);
            // Restore a placeholder card so the pane remains visible on the dashboard.
            if let Some((pane_id, cwd)) = pane_id_and_cwd
                && self.managed_pane_ids.contains(&pane_id)
            {
                self.insert_placeholder_session(pane_id, cwd);
            }
            return;
        }

        let pane_started = event
            .pane_id
            .as_ref()
            .and_then(|pid| self.pane_started_at.get(pid))
            .copied();

        let session = self
            .sessions
            .entry(event.session_id.clone())
            .or_insert_with(|| SessionState {
                session_id: event.session_id.clone(),
                agent_type: event.agent_type.clone(),
                cwd: event.cwd.clone(),
                status: SessionStatus::Idle,
                active_tool: None,
                started_at: pane_started.unwrap_or(event.timestamp),
                last_activity: event.timestamp,
                recent_events: VecDeque::new(),
                tool_count: 0,
                last_user_prompt: None,
                first_prompts: Vec::new(),
                pane_id: event.pane_id.clone(),
                git_branch: None,
                git_branch_refreshed_at: None,
                git_worktree: None,
            });

        session.last_activity = event.timestamp;

        if session.agent_type == AgentType::None && event.agent_type != AgentType::None {
            session.agent_type = event.agent_type.clone();
        }

        if event.cwd.is_some() {
            session.cwd.clone_from(&event.cwd);
        }

        if let Some(ref prompt) = event.user_prompt {
            session.last_user_prompt = Some(prompt.clone());
            if session.first_prompts.len() < MAX_FIRST_PROMPTS {
                session.first_prompts.push(prompt.clone());
            }
        }

        if event.pane_id.is_some() {
            session.pane_id.clone_from(&event.pane_id);
        }

        match event.event_type {
            EventType::SessionStart => {
                session.status = SessionStatus::Idle;
                session.active_tool = None;
            }
            EventType::Thinking => {
                session.status = SessionStatus::Thinking;
                session.active_tool = None;
            }
            EventType::ToolStart => {
                let tool_name = event.tool_name.clone().unwrap_or_default();
                if session.status != SessionStatus::WaitingForInput {
                    session.status = if Self::is_interactive_tool(&tool_name) {
                        SessionStatus::WaitingForInput
                    } else {
                        SessionStatus::Working
                    };
                }
                session.active_tool = Some(ActiveTool {
                    name: tool_name,
                    detail: event.tool_detail.clone(),
                });
            }
            EventType::ToolEnd => {
                session.active_tool = None;
                session.tool_count += 1;
                if session.status == SessionStatus::WaitingForInput {
                    session.status = SessionStatus::Thinking;
                }
            }
            EventType::WaitingForInput | EventType::PermissionRequest => {
                session.status = SessionStatus::WaitingForInput;
            }
            EventType::Idle => {
                session.status = SessionStatus::Idle;
                session.active_tool = None;
            }
            EventType::Compacting => {
                session.status = SessionStatus::Compacting;
                session.active_tool = None;
            }
            EventType::SubagentStart | EventType::SubagentStop => {
                // Informational — recorded in recent_events but no status change
            }
            EventType::Error => {
                session.status = SessionStatus::Error;
            }
            EventType::SessionEnd => unreachable!(),
        }

        session.recent_events.push_back(event);
        if session.recent_events.len() > MAX_RECENT_EVENTS {
            session.recent_events.pop_front();
        }
    }
}

/// Refresh the git branch for a session.
/// Runs `git -C <cwd> rev-parse --abbrev-ref HEAD` synchronously.
/// Falls back to directory basename if cwd is not a git repo.
/// If cwd is None, marks the session as attempted so it won't be retried every tick.
pub fn refresh_git_branch(session: &mut SessionState) {
    let now = Utc::now();
    let cwd = match &session.cwd {
        Some(c) => c.clone(),
        None => {
            // Mark as attempted so we don't retry every tick.
            session.git_branch_refreshed_at = Some(now);
            return;
        }
    };

    let branch = run_git_branch(&cwd);
    session.git_branch = Some(branch);
    session.git_worktree = run_git_worktree_name(&cwd);
    session.git_branch_refreshed_at = Some(now);
}

pub(crate) fn run_git_branch(cwd: &str) -> String {
    use std::process::Command;

    let output = Command::new("git")
        .args(["-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if branch == "HEAD" {
                // Detached HEAD — get short SHA
                let sha_out = Command::new("git")
                    .args(["-C", cwd, "rev-parse", "--short", "HEAD"])
                    .output();
                match sha_out {
                    Ok(s) if s.status.success() => {
                        format!("detached@{}", String::from_utf8_lossy(&s.stdout).trim())
                    }
                    _ => "detached".to_string(),
                }
            } else {
                branch
            }
        }
        _ => {
            // Not a git repo — use dir basename
            std::path::Path::new(cwd)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| cwd.to_string())
        }
    }
}

/// Detect whether `cwd` is inside a linked git worktree.
/// Returns the worktree's directory name when `cwd` is in a linked worktree,
/// or `None` for the main checkout / non-git directories.
///
/// Git stores linked worktrees at `<common>/worktrees/<name>`, so when
/// `--absolute-git-dir` differs from `--git-common-dir` we are in a linked
/// worktree and the worktree name is the basename of `--absolute-git-dir`.
pub(crate) fn run_git_worktree_name(cwd: &str) -> Option<String> {
    use std::process::Command;

    let run = |arg: &str| {
        Command::new("git")
            .args(["-C", cwd, "rev-parse", arg])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };

    let git_dir = run("--absolute-git-dir")?;
    let common_dir = run("--git-common-dir")?;

    let git_dir_canon = std::fs::canonicalize(&git_dir).ok()?;
    let common_canon = std::fs::canonicalize(&common_dir).ok()?;

    if git_dir_canon == common_canon {
        return None;
    }

    git_dir_canon
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{AgentEvent, AgentType, EventType};
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_event(session_id: &str, event_type: EventType) -> AgentEvent {
        AgentEvent {
            session_id: session_id.to_string(),
            agent_type: AgentType::ClaudeCode,
            event_type,
            tool_name: None,
            tool_detail: None,
            cwd: Some("/tmp".to_string()),
            timestamp: Utc::now(),
            user_prompt: None,
            metadata: HashMap::new(),
            pane_id: None,
        }
    }

    #[test]
    fn full_session_lifecycle() {
        let mut state = AppState::default();

        state.apply_event(make_event("s1", EventType::SessionStart));
        assert_eq!(state.sessions["s1"].status, SessionStatus::Idle);

        let mut tool_event = make_event("s1", EventType::ToolStart);
        tool_event.tool_name = Some("Read".to_string());
        tool_event.tool_detail = Some("main.rs".to_string());
        state.apply_event(tool_event);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
        assert_eq!(
            state.sessions["s1"].active_tool.as_ref().unwrap().name,
            "Read"
        );

        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
        assert!(state.sessions["s1"].active_tool.is_none());

        state.apply_event(make_event("s1", EventType::SessionEnd));
        assert!(!state.sessions.contains_key("s1"));
    }

    #[test]
    fn concurrent_sessions() {
        let mut state = AppState::default();

        state.apply_event(make_event("s1", EventType::SessionStart));
        state.apply_event(make_event("s2", EventType::SessionStart));
        assert_eq!(state.sessions.len(), 2);

        let mut tool_event = make_event("s1", EventType::ToolStart);
        tool_event.tool_name = Some("Write".to_string());
        state.apply_event(tool_event);

        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
        assert_eq!(state.sessions["s2"].status, SessionStatus::Idle);
    }

    #[test]
    fn reuse_session_for_same_pane() {
        let mut state = AppState::default();
        state.register_pane("pane-1".to_string());

        let mut first = make_event("s1", EventType::SessionStart);
        first.pane_id = Some("pane-1".to_string());
        state.apply_event(first);

        let mut restart = make_event("s2", EventType::SessionStart);
        restart.pane_id = Some("pane-1".to_string());
        state.apply_event(restart);

        assert!(state.sessions.contains_key("s1"));
        assert!(!state.sessions.contains_key("s2"));
        assert_eq!(state.sessions["s1"].pane_id.as_deref(), Some("pane-1"));
    }

    #[test]
    fn auto_create_unknown_session() {
        let mut state = AppState::default();

        let mut tool_event = make_event("unknown", EventType::ToolStart);
        tool_event.tool_name = Some("Bash".to_string());
        state.apply_event(tool_event);

        assert!(state.sessions.contains_key("unknown"));
        assert_eq!(state.sessions["unknown"].status, SessionStatus::Working);
    }

    #[test]
    fn event_buffer_capping() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        for _ in 0..60 {
            state.apply_event(make_event("s1", EventType::Idle));
        }

        // 1 SessionStart + 60 Idle = 61, capped to 50
        assert_eq!(state.sessions["s1"].recent_events.len(), 50);
    }

    #[test]
    fn waiting_for_input_status() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        state.apply_event(make_event("s1", EventType::WaitingForInput));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
        assert!(state.sessions["s1"].active_tool.is_none());
    }

    #[test]
    fn notification_during_active_tool_shows_waiting() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Bash".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);

        // A Notification during an active tool means a permission prompt —
        // PreToolUse fires before the Notification, so active_tool is set.
        state.apply_event(make_event("s1", EventType::WaitingForInput));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
        assert!(state.sessions["s1"].active_tool.is_some());
    }

    #[test]
    fn ask_user_question_shows_waiting_for_input() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("AskUserQuestion".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);

        // Notification follow-up is idempotent — status stays WaitingForInput.
        state.apply_event(make_event("s1", EventType::WaitingForInput));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
    }

    #[test]
    fn exit_plan_mode_shows_waiting_for_input() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("ExitPlanMode".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
    }

    #[test]
    fn regular_tool_still_shows_working() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Read".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);

        let mut tool_start_bash = make_event("s1", EventType::ToolStart);
        tool_start_bash.tool_name = Some("Bash".to_string());
        state.apply_event(tool_start_bash);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
    }

    #[test]
    fn tool_count_increments_on_tool_end() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        assert_eq!(state.sessions["s1"].tool_count, 0);

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Read".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].tool_count, 0);

        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(state.sessions["s1"].tool_count, 1);

        let mut tool_start2 = make_event("s1", EventType::ToolStart);
        tool_start2.tool_name = Some("Write".to_string());
        state.apply_event(tool_start2);
        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(state.sessions["s1"].tool_count, 2);
    }

    #[test]
    fn tool_end_clears_waiting_for_input() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        // Simulate: PreToolUse → PermissionRequest → tool runs → PostToolUse
        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Bash".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);

        state.apply_event(make_event("s1", EventType::PermissionRequest));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);

        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(state.sessions["s1"].status, SessionStatus::Thinking);
    }

    #[test]
    fn toolstart_does_not_override_waiting_for_input() {
        // Regression: a concurrent subagent firing PreToolUse while a permission
        // prompt is active must not knock the status back to Working.
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        state.apply_event(make_event("s1", EventType::PermissionRequest));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);

        let mut subagent_tool = make_event("s1", EventType::ToolStart);
        subagent_tool.tool_name = Some("Explore".to_string());
        state.apply_event(subagent_tool);
        assert_eq!(
            state.sessions["s1"].status,
            SessionStatus::WaitingForInput,
            "ToolStart must not override WaitingForInput"
        );
        assert_eq!(
            state.sessions["s1"]
                .active_tool
                .as_ref()
                .map(|t| t.name.as_str()),
            Some("Explore"),
            "active_tool must still be updated even when status is preserved"
        );
    }

    #[test]
    fn toolstart_sets_working_when_not_waiting() {
        // Normal flow: ToolStart should still set Working when no permission prompt.
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Bash".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
        assert_eq!(
            state.sessions["s1"]
                .active_tool
                .as_ref()
                .map(|t| t.name.as_str()),
            Some("Bash"),
            "active_tool must be set on normal ToolStart"
        );
    }

    #[test]
    fn tool_end_preserves_working_status() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Bash".to_string());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);

        // ToolEnd without permission request should keep Working→Working (not change)
        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(state.sessions["s1"].status, SessionStatus::Working);
    }

    #[test]
    fn error_status() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        state.apply_event(make_event("s1", EventType::Error));
        assert_eq!(state.sessions["s1"].status, SessionStatus::Error);
    }

    #[test]
    fn last_user_prompt_set_and_persists() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        assert!(state.sessions["s1"].last_user_prompt.is_none());

        let mut prompt_event = make_event("s1", EventType::Thinking);
        prompt_event.user_prompt = Some("fix the bug".to_string());
        state.apply_event(prompt_event);
        assert_eq!(
            state.sessions["s1"].last_user_prompt.as_deref(),
            Some("fix the bug")
        );

        // Subsequent event without prompt should not clear it
        state.apply_event(make_event("s1", EventType::ToolEnd));
        assert_eq!(
            state.sessions["s1"].last_user_prompt.as_deref(),
            Some("fix the bug")
        );

        // New prompt replaces old one
        let mut prompt_event2 = make_event("s1", EventType::Thinking);
        prompt_event2.user_prompt = Some("add tests".to_string());
        state.apply_event(prompt_event2);
        assert_eq!(
            state.sessions["s1"].last_user_prompt.as_deref(),
            Some("add tests")
        );
    }

    #[test]
    fn first_prompts_captures_up_to_three() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        assert!(state.sessions["s1"].first_prompts.is_empty());

        let prompts = ["first", "second", "third"];
        for (i, text) in prompts.iter().enumerate() {
            let mut ev = make_event("s1", EventType::Thinking);
            ev.user_prompt = Some(text.to_string());
            state.apply_event(ev);
            assert_eq!(state.sessions["s1"].first_prompts.len(), i + 1);
            assert_eq!(state.sessions["s1"].first_prompts[i], *text);
        }
    }

    #[test]
    fn first_prompts_no_overwrite_after_cap() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        for text in &["p1", "p2", "p3", "p4", "p5"] {
            let mut ev = make_event("s1", EventType::Thinking);
            ev.user_prompt = Some(text.to_string());
            state.apply_event(ev);
        }

        assert_eq!(state.sessions["s1"].first_prompts.len(), 3);
        assert_eq!(state.sessions["s1"].first_prompts[0], "p1");
        assert_eq!(state.sessions["s1"].first_prompts[1], "p2");
        assert_eq!(state.sessions["s1"].first_prompts[2], "p3");
    }

    #[test]
    fn first_prompts_persist_across_events() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));

        let mut ev = make_event("s1", EventType::Thinking);
        ev.user_prompt = Some("only prompt".to_string());
        state.apply_event(ev);

        state.apply_event(make_event("s1", EventType::ToolEnd));
        state.apply_event(make_event("s1", EventType::Idle));
        state.apply_event(make_event("s1", EventType::Thinking));

        assert_eq!(state.sessions["s1"].first_prompts.len(), 1);
        assert_eq!(state.sessions["s1"].first_prompts[0], "only prompt");
    }

    #[test]
    fn aggregate_stats_empty() {
        let state = AppState::default();
        let stats = state.aggregate_stats();
        assert_eq!(stats, DashboardStats::default());
    }

    #[test]
    fn aggregate_stats_mixed_sessions() {
        let mut state = AppState::default();

        state.apply_event(make_event("s1", EventType::SessionStart));
        let mut tool = make_event("s1", EventType::ToolStart);
        tool.tool_name = Some("Read".to_string());
        state.apply_event(tool);
        // s1: Working

        state.apply_event(make_event("s2", EventType::SessionStart));
        state.apply_event(make_event("s2", EventType::WaitingForInput));
        // s2: WaitingForInput

        state.apply_event(make_event("s3", EventType::SessionStart));
        state.apply_event(make_event("s3", EventType::Error));
        // s3: Error

        state.apply_event(make_event("s4", EventType::SessionStart));
        state.apply_event(make_event("s4", EventType::Thinking));
        // s4: Thinking

        state.apply_event(make_event("s5", EventType::SessionStart));
        // s5: Idle

        let stats = state.aggregate_stats();
        assert_eq!(stats.active, 5);
        assert_eq!(stats.working, 1);
        assert_eq!(stats.waiting, 1);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.thinking, 1);
        assert_eq!(stats.idle, 1);
    }

    #[test]
    fn aggregate_stats_tool_count_summation() {
        let mut state = AppState::default();

        state.apply_event(make_event("s1", EventType::SessionStart));
        let mut t1 = make_event("s1", EventType::ToolStart);
        t1.tool_name = Some("Read".to_string());
        state.apply_event(t1);
        state.apply_event(make_event("s1", EventType::ToolEnd));

        state.apply_event(make_event("s2", EventType::SessionStart));
        for _ in 0..3 {
            let mut t = make_event("s2", EventType::ToolStart);
            t.tool_name = Some("Bash".to_string());
            state.apply_event(t);
            state.apply_event(make_event("s2", EventType::ToolEnd));
        }

        let stats = state.aggregate_stats();
        assert_eq!(stats.total_tools, 4);
    }

    #[test]
    fn restarted_session_preserves_started_at_via_pane() {
        let mut state = AppState::default();
        state.register_pane("pane-42".to_string());

        // Register session with a pane
        let mut ev = make_event("s1", EventType::SessionStart);
        ev.pane_id = Some("pane-42".to_string());
        state.apply_event(ev);
        let original_started = state.sessions["s1"].started_at;

        // End the session (simulates /clear)
        let mut end_ev = make_event("s1", EventType::SessionEnd);
        end_ev.pane_id = Some("pane-42".to_string());
        state.apply_event(end_ev);
        // After SessionEnd, a placeholder is restored since the pane is still managed.
        // Key is "pane-pane-42" because pane_id="pane-42" and placeholder keys use "pane-{pane_id}".
        assert!(state.sessions.contains_key("pane-pane-42"));

        // New session on the same pane reuses the placeholder key and keeps started_at.
        let mut ev2 = make_event("s2", EventType::SessionStart);
        ev2.pane_id = Some("pane-42".to_string());
        state.apply_event(ev2);
        assert_eq!(state.sessions["pane-pane-42"].started_at, original_started);
    }

    #[test]
    fn permission_request_sets_waiting_for_input() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        state.apply_event(make_event("s1", EventType::PermissionRequest));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
    }

    #[test]
    fn tool_start_preserves_waiting_for_input() {
        let mut state = AppState::default();
        state.apply_event(make_event("s1", EventType::SessionStart));
        state.apply_event(make_event("s1", EventType::WaitingForInput));
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);

        let mut tool_start = make_event("s1", EventType::ToolStart);
        tool_start.tool_name = Some("Bash".into());
        state.apply_event(tool_start);
        assert_eq!(state.sessions["s1"].status, SessionStatus::WaitingForInput);
    }

    #[test]
    fn placeholder_session_created() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        assert!(state.sessions.contains_key("pane-42"));
        let session = &state.sessions["pane-42"];
        assert_eq!(session.agent_type, AgentType::None);
        assert_eq!(session.status, SessionStatus::Idle);
        assert_eq!(session.pane_id.as_deref(), Some("42"));
        assert_eq!(session.cwd.as_deref(), Some("/tmp"));
        assert_eq!(session.tool_count, 0);
    }

    #[test]
    fn placeholder_transitions_to_real_session() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        let mut start = make_event("real-uuid-123", EventType::SessionStart);
        start.pane_id = Some("42".to_string());
        start.cwd = Some("/home".to_string());
        state.apply_event(start);

        // Placeholder key is reused, real UUID key is removed
        assert!(state.sessions.contains_key("pane-42"));
        assert!(!state.sessions.contains_key("real-uuid-123"));
        let session = &state.sessions["pane-42"];
        assert_eq!(session.agent_type, AgentType::ClaudeCode);
        assert_eq!(session.cwd.as_deref(), Some("/home"));
        assert_eq!(session.pane_id.as_deref(), Some("42"));
    }

    #[test]
    fn placeholder_restored_after_session_end() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        // Transition to real session
        let mut start = make_event("real-uuid", EventType::SessionStart);
        start.pane_id = Some("42".to_string());
        state.apply_event(start);
        assert_eq!(state.sessions["pane-42"].agent_type, AgentType::ClaudeCode);

        // End the real session — placeholder should be restored
        let mut end = make_event("pane-42", EventType::SessionEnd);
        end.pane_id = Some("42".to_string());
        state.apply_event(end);

        assert!(state.sessions.contains_key("pane-42"));
        assert_eq!(state.sessions["pane-42"].agent_type, AgentType::None);
        assert_eq!(state.sessions["pane-42"].pane_id.as_deref(), Some("42"));
    }

    #[test]
    fn placeholder_not_restored_after_close() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        // Transition to real session
        let mut start = make_event("real-uuid", EventType::SessionStart);
        start.pane_id = Some("42".to_string());
        state.apply_event(start);

        // Simulate Ctrl+w: remove session and unregister pane (same as ui handler)
        state.sessions.remove("pane-42");
        state.unregister_pane("42");

        assert!(state.sessions.is_empty());
        assert!(!state.managed_pane_ids.contains("42"));
    }

    #[test]
    fn placeholder_excluded_from_stats() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        // Add a real session on a different registered pane
        state.register_pane("99".to_string());
        let mut start = make_event("s1", EventType::SessionStart);
        start.pane_id = Some("99".to_string());
        state.apply_event(start);

        let stats = state.aggregate_stats();
        assert_eq!(stats.active, 1);
        assert_eq!(stats.idle, 1);
    }

    #[test]
    fn close_placeholder_session() {
        let mut state = AppState::default();
        state.register_pane("42".to_string());
        state.insert_placeholder_session("42".to_string(), Some("/tmp".to_string()));

        // Simulate Ctrl+w on the placeholder
        state.sessions.remove("pane-42");
        state.unregister_pane("42");

        assert!(state.sessions.is_empty());
        assert!(!state.managed_pane_ids.contains("42"));
    }

    #[test]
    fn handle_delegate_stores_event() {
        let mut state = AppState::default();
        state
            .pane_role_map
            .insert("pane-1".into(), "orchestrator".into());
        state.orchestrator_pane_ids.insert("pane-1".into());

        let signal = crate::event::DelegateSignal {
            pane_id: "pane-1".into(),
            task: "Implement login".into(),
            to: vec!["coder".into()],
            timestamp: Utc::now(),
        };
        state.handle_delegate(signal);

        assert_eq!(state.delegate_events.len(), 1);
        assert_eq!(state.delegate_events[0].task, "Implement login");
        assert_eq!(state.delegate_events[0].to, vec!["coder"]);
    }

    #[test]
    fn handle_delegate_unknown_pane_is_noop() {
        let mut state = AppState::default();

        let signal = crate::event::DelegateSignal {
            pane_id: "unknown-pane".into(),
            task: "Do something".into(),
            to: vec!["coder".into()],
            timestamp: Utc::now(),
        };
        state.handle_delegate(signal);

        assert!(state.delegate_events.is_empty());
    }

    #[test]
    fn handle_work_done_resolves_role_and_stores_event() {
        let mut state = AppState::default();
        state.pane_role_map.insert("pane-1".into(), "coder".into());
        state
            .pane_cwd_map
            .insert("pane-1".into(), "/tmp/test-wd".into());

        let signal = crate::event::WorkDoneSignal {
            pane_id: "pane-1".into(),
            task: "Implemented login".into(),
            done: false,
            timestamp: Utc::now(),
        };
        state.handle_work_done(signal);

        assert_eq!(state.work_done_events.len(), 1);
        assert_eq!(state.work_done_events[0].task, "Implemented login");

        // Verify summary file was written
        let file = std::path::Path::new("/tmp/test-wd/.dot-agent-deck/work-done-coder.md");
        assert!(file.exists());
        let content = std::fs::read_to_string(file).unwrap();
        assert_eq!(content, "Implemented login");

        // Clean up
        let _ = std::fs::remove_dir_all("/tmp/test-wd/.dot-agent-deck");
    }

    #[test]
    fn handle_work_done_unknown_pane_is_noop() {
        let mut state = AppState::default();

        let signal = crate::event::WorkDoneSignal {
            pane_id: "unknown-pane".into(),
            task: "Some work".into(),
            done: false,
            timestamp: Utc::now(),
        };
        state.handle_work_done(signal);

        assert!(state.work_done_events.is_empty());
    }

    #[test]
    fn handle_work_done_done_flag_stored() {
        let mut state = AppState::default();
        state
            .pane_role_map
            .insert("pane-1".into(), "orchestrator".into());

        let signal = crate::event::WorkDoneSignal {
            pane_id: "pane-1".into(),
            task: "All complete".into(),
            done: true,
            timestamp: Utc::now(),
        };
        state.handle_work_done(signal);

        assert_eq!(state.work_done_events.len(), 1);
        assert!(state.work_done_events[0].done);
    }

    #[test]
    fn refresh_git_branch_non_git_dir() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_string_lossy().to_string();
        let basename = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let mut session = SessionState {
            session_id: "test".to_string(),
            agent_type: crate::event::AgentType::ClaudeCode,
            cwd: Some(path),
            status: SessionStatus::Idle,
            active_tool: None,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            recent_events: VecDeque::new(),
            tool_count: 0,
            last_user_prompt: None,
            first_prompts: Vec::new(),
            pane_id: None,
            git_branch: None,
            git_branch_refreshed_at: None,
            git_worktree: None,
        };

        refresh_git_branch(&mut session);

        assert_eq!(session.git_branch.as_deref(), Some(basename.as_str()));
    }

    #[test]
    fn refresh_git_branch_sets_refreshed_at() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        let mut session = SessionState {
            session_id: "test".to_string(),
            agent_type: crate::event::AgentType::ClaudeCode,
            cwd: Some(path),
            status: SessionStatus::Idle,
            active_tool: None,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            recent_events: VecDeque::new(),
            tool_count: 0,
            last_user_prompt: None,
            first_prompts: Vec::new(),
            pane_id: None,
            git_branch: None,
            git_branch_refreshed_at: None,
            git_worktree: None,
        };

        assert!(session.git_branch_refreshed_at.is_none());
        refresh_git_branch(&mut session);
        assert!(session.git_branch_refreshed_at.is_some());
    }

    #[test]
    fn refresh_git_branch_cwd_none_sets_refreshed_at_without_branch() {
        let mut session = SessionState {
            session_id: "test-none-cwd".to_string(),
            agent_type: crate::event::AgentType::None,
            cwd: None,
            status: SessionStatus::Idle,
            active_tool: None,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            recent_events: VecDeque::new(),
            tool_count: 0,
            last_user_prompt: None,
            first_prompts: Vec::new(),
            pane_id: None,
            git_branch: None,
            git_branch_refreshed_at: None,
            git_worktree: None,
        };

        assert!(session.git_branch_refreshed_at.is_none());
        refresh_git_branch(&mut session);
        // Should be marked attempted so refresh_stale_branches won't re-check every tick.
        assert!(session.git_branch_refreshed_at.is_some());
        // Branch remains None since there is no cwd to inspect.
        assert!(session.git_branch.is_none());
    }

    #[test]
    fn run_git_worktree_name_main_checkout_is_none() {
        let repo_root = env!("CARGO_MANIFEST_DIR");
        assert_eq!(run_git_worktree_name(repo_root), None);
    }

    #[test]
    fn run_git_worktree_name_non_git_dir_is_none() {
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_string_lossy().to_string();
        assert_eq!(run_git_worktree_name(&path), None);
    }

    #[test]
    fn run_git_worktree_name_linked_worktree_returns_basename() {
        use std::process::Command;
        use tempfile::TempDir;

        let source_repo = env!("CARGO_MANIFEST_DIR");
        let tmp = TempDir::new().unwrap();
        let wt_path = tmp.path().join("dad-wt-unit-test");
        let branch_name = format!(
            "dad-wt-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let add = Command::new("git")
            .args([
                "-C",
                source_repo,
                "worktree",
                "add",
                "-b",
                &branch_name,
                wt_path.to_str().unwrap(),
            ])
            .output()
            .expect("spawn git worktree add");
        if !add.status.success() {
            panic!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&add.stderr)
            );
        }

        let result = run_git_worktree_name(wt_path.to_str().unwrap());

        // Clean up worktree and branch regardless of assertion outcome.
        let _ = Command::new("git")
            .args([
                "-C",
                source_repo,
                "worktree",
                "remove",
                "--force",
                wt_path.to_str().unwrap(),
            ])
            .output();
        let _ = Command::new("git")
            .args(["-C", source_repo, "branch", "-D", &branch_name])
            .output();

        assert_eq!(result.as_deref(), Some("dad-wt-unit-test"));
    }

    #[test]
    fn refresh_git_branch_populates_worktree_on_main_checkout() {
        let repo_root = env!("CARGO_MANIFEST_DIR").to_string();
        let mut session = SessionState {
            session_id: "wt-branch".to_string(),
            agent_type: crate::event::AgentType::None,
            cwd: Some(repo_root),
            status: SessionStatus::Idle,
            active_tool: None,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            recent_events: VecDeque::new(),
            tool_count: 0,
            last_user_prompt: None,
            first_prompts: Vec::new(),
            pane_id: None,
            git_branch: None,
            git_branch_refreshed_at: None,
            git_worktree: None,
        };

        refresh_git_branch(&mut session);
        assert!(session.git_branch.is_some());
        assert!(session.git_branch_refreshed_at.is_some());
        assert!(session.git_worktree.is_none());
    }
}
