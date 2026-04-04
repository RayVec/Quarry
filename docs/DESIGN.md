# QUARRY Design System

## The Architectural Intelligence Framework

QUARRY's interface is guided by **The Digital Blueprint**.

The product should feel like a precision research instrument rather than a generic SaaS dashboard. The visual language is intentionally closer to an editorial technical journal crossed with an architectural drafting environment:

- **Intentional asymmetry:** the main workspace is offset and leaves a wide right margin so dense research output feels placed, not boxed.
- **Tonal authority:** layout is defined by surface shifts and negative space more than by borders.
- **Data as art:** citations, verification states, and research status indicators are treated as premium visual signals rather than incidental badges.

## Core Principles

### 1. The No-Line Rule

QUARRY should not rely on heavy 1px dividers to create structure.

Prefer:

- background transitions
- tonal lift
- spacing voids
- left accent bars for active or verified states

Only use a ghost border when contrast would otherwise become unclear. A ghost border should be low-opacity and barely perceptible.

### 2. Tonal Layering

The UI is built as stacked surfaces:

1. **Base workspace:** `surface` for the main canvas
2. **Navigation/sidebar:** `surface-container-low`
3. **Raised modules:** `surface-container-lowest`
4. **Active or overlay states:** `surface-container-high`

Standard cards should not rely on thick borders or strong shadows. Lift should come primarily from tone and only secondarily from very soft ambient shadow.

### 3. Editorial Typography

QUARRY uses a dual-font system:

- **Manrope** for display text and section-level headings
- **Inter** for body text, labels, metadata, and dense information

Hierarchy should consistently pair:

- a **label** or **eyebrow** for category/context
- a stronger **headline** for the actual meaning

Body copy should avoid pure black and instead use a deep ink tone to preserve an ink-on-paper feel.

### 4. Rectilinear Inputs and Controls

Inputs and controls should feel engineered rather than consumer-soft:

- rectangular or softly rounded shapes
- low-contrast full outlines for fields
- solid-fill primary actions
- muted secondary actions for supporting workflows
- no capsule-style pill buttons for major actions

The default control set should map to four visual treatments:

- **Primary:** filled `primary` for the dominant action
- **Secondary:** filled `secondary` for supporting actions on pale surfaces
- **Inverted:** dark neutral fill for reversed or utility emphasis
- **Outlined:** pale background with an `outline-variant` stroke

### 5. Verification as a First-Class Signal

Verification state is the hero of the system.

Use:

- deep green / mint tonal cues for verified states
- amber for warnings
- red-tinted surfaces for unverified or removed claims
- narrow vertical accent bars for sentence status

These signals should feel deliberate and restrained, not gamified.

## Layout System

### Global Shell

The default application shell is a two-column layout:

- **Left sidebar:** persistent navigation and recent activity
- **Right workspace:** the live research surface

The sidebar contains:

- QUARRY brand block
- `New Search`
- `Recent Research`

The workspace contains:

- an optional light utility bar in landing mode
- either a landing hero or the live conversation thread
- a docked query composer

### Landing State

The empty state should feel like a clean research launch surface:

- strong display headline
- short supporting paragraph
- one primary query composer

The user should understand immediately that QUARRY is for high-precision document research, not casual chat.

### Research Thread State

Once a query exists, the layout transitions into a working research surface:

- the latest query is surfaced as the thread title
- user messages appear as quiet raised modules
- assistant messages are primarily content-first, not boxed
- the composer remains docked at the bottom of the workspace

## Component Rules

### Sidebar

- uses `surface-container-low`
- does not use explicit divider lines between sections
- recent items use tonal selection rather than outlined cards

### Query Composer

- uses `surface-container-lowest`
- starts as a single-line composer
- grows with input
- caps visible content height after seven lines of text
- places the submit action inline for one-line input and on its own row for multiline input

### User Message

- may appear as a raised module on `surface-container-lowest`
- includes a left accent bar to show the user-originated prompt

### Assistant Message

- should avoid a generic outer card wrapper
- should read like a clean editorial document surface
- internal review modules may still use tonal lift where needed

### Citation Tags

- use small rectilinear tags, not pills
- should feel like compact technical annotations

### Review Panel

- uses tonal lift rather than section dividers
- summary and actions should be grouped through spacing and background

### Drawers

- use glassmorphism with blurred `surface-variant`
- should slide over the workspace rather than reflow content

### Icons

- use `lucide-react` as the default icon library for interface controls
- keep icon style consistent: line icons, regular stroke weight, and restrained sizing
- utility actions (for example diagnostics, drawer controls, and inline review tools) should prefer icon-only buttons with accessible labels

## Color Tokens

The design system now anchors itself to the palette shown in the card reference:

- `primary`: `#1B365D`
- `secondary`: `#475569`
- `tertiary`: `#059669`
- `neutral`: `#0F172A`

These colors are used semantically:

- `primary` drives the main action, active navigation, and dominant data accents
- `secondary` handles body copy, supporting controls, metadata, and secondary actions
- `tertiary` signals verified, grounded, or positive states
- `neutral` anchors headlines, dense ink, and high-contrast iconography

The surrounding surfaces should sit in the near-white end of the `secondary` scale rather than read as blue-purple:

- `surface`: `#F8FAFC`
- `surface-container-low`: `#F1F5F9`
- `surface-container`: `#E9EEF5`
- `surface-container-lowest`: `#FFFFFF`
- `surface-container-high`: `#E2E8F0`
- `surface-container-highest`: `#D8E0EA`
- `surface-dim`: `#ECF1F6`
- `surface-variant`: translucent lift derived from white and the lightest `secondary` tint

Supporting tokens in the current implementation:

- `primary-container`: darker companion for hover and pressed primary states
- `primary-fixed`: pale blueprint tint for annotations and passive highlights
- `outline-variant`: low-contrast structural line for fields and outlined controls
- `tertiary-fixed`: soft verified-state fill
- `warning-surface` / `warning-ink`: restrained amber review feedback
- `error-surface` / `error-ink`: restrained red review feedback
- `on-surface`: defaults to `neutral`
- `muted`: defaults to `secondary`

## Interaction Notes

- hover states should sharpen tone, not bounce dramatically
- standard shadows should stay subtle and tinted by the surrounding surface
- mobile layout should preserve the sidebar information architecture by stacking rather than dropping important navigation affordances
- the system should always prefer clarity and research trust over decorative UI density

## Current Implementation Notes

The current frontend implementation applies this system in:

- `web/src/App.tsx`
- `web/src/components/QueryComposer.tsx`
- `web/src/components/ConversationMessage.tsx`
- `web/src/components/PendingConversationMessage.tsx`
- `web/src/components/ResponseReview.tsx`
- `web/src/components/ReviewPanel.tsx`
- `web/src/components/CitationDialog.tsx`
- `web/src/components/DiagnosticsDrawer.tsx`
- `web/src/styles/app.css`

The design system is intended to be the source of truth for future UI revisions. New surfaces should conform to this tonal, editorial, and no-line approach unless there is a strong product reason to deviate.
