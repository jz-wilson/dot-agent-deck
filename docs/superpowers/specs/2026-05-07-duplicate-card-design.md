# Duplicate Card Feature Design

## Goal
Add `Ctrl+Y` in Normal mode (or any mode where the dashboard is visible) to instantly
duplicate the selected card as a new live pane in the same repo.

## Keybinding
`Ctrl+Y` — free, not taken by any existing binding.
Note: `Ctrl+D` is taken (exit PaneInput → Normal mode / dashboard toggle).

## Behavior
- Same dir, name, command, mode_config as source pane
- If source pane had a mode set, mode_config is reconstructed via load_project_config()
- Git branch in footer differentiates cards with same name
- All guard cases (no pane_id, missing metadata, empty list) silently skip

## Implementation
- `find_duplicate_source(index, filtered, pane_metadata) -> Option<&SavedPane>` — pure, testable
- Ctrl+Y arm in global Ctrl shortcuts block (~line 2867 src/ui.rs)
- `duplicate_req: Option<NewPaneRequest>` local var fed into `let result =` at line ~3177
- Reuses existing `KeyResult::NewPane` processing path — no code duplication

## Test behaviors (4 unit tests)
1. Happy path: selected session has pane_id + metadata entry → returns Some(&SavedPane)
2. No pane_id guard: session.pane_id = None → None
3. Missing metadata guard: pane_id present but not in pane_metadata → None
4. Bounds guard: selected_index >= filtered.len() → None
