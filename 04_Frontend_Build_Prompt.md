# TriagePlus — Frontend Build Prompt

**Project:** TriagePlus · IIT Dharwad Summer of Innovation · Team "Hardly Human"
**Depends on:** `02_Backend_Architecture_Build_Prompt.md` (WebSocket message types, REST routes, public endpoints), `01_Database_Schema_Build_Prompt.md` (field names — use `triage_level` and `confidence`, not "severity"/"urgency")

## 1. Overview

TriagePlus's frontend is a single, mobile-first React SPA serving two audiences with two distinct design languages: a calming, conversational patient-facing app (`/`, `/chat`), and a clinical, high-density Doctor Portal (`/doctor/*`), plus an internal developer diagnostics view (`/diagnostics`). It communicates with the backend primarily over WebSockets for the chat experience and over REST for the Doctor Portal. Every screen must be built and tested mobile-first, not adapted afterward — most patients will open this on a phone.

## 2. Tech Stack

React 19, Vite 6, TypeScript, Zustand, React Router DOM v7, Tailwind CSS 3 + Autoprefixer + PostCSS, Lucide-React icons. Use `Suspense` + lazy loading per route.

## 3. Routing

| Route | Purpose |
|---|---|
| `/` | Landing page, starts a new triage chat |
| `/chat` | Core WebSocket triage session — `ws/chat/{session_id}` |
| `/doctor/login` | Doctor auth via Supabase Auth |
| `/doctor/dashboard` and other `/doctor/*` | Doctor Portal (§7) |
| `/diagnostics` | RAG/LangGraph developer monitor — `ws/diagnostics` |

## 4. Design System — Patient-Facing (`/`, `/chat`)

A nature-inspired, calming theme with full light/dark support via Tailwind's `class` strategy (`darkMode: 'class'` in `tailwind.config.js`).

**Light mode:**
- Primary: `canopy-green` (#0a3922), `coral-pulse` (#ff643b), `leaf-bright` (#1dbf73)
- Accents: `indigo-bloom`, `deep-teal`, `sky-signal` (#00b4dc), `orchid-tint`
- Background/surface: `paper-white`
- Text: `ink-black`, muted text `charcoal` (#333333)
- Triage semantic colors: `triage-red`, `triage-orange`, `triage-yellow`, `triage-green`

**Dark mode** (define these as real Tailwind color tokens, not an afterthought filter over the light palette):
- Background: near-black warm tone, e.g. `#0b120e` (`ink-black` deepened)
- Surface (cards, chat bubbles): `charcoal` (#333333) / `graphite`
- Text: `paper-white` primary, `cloud-gray` for secondary/muted text
- Primary accents unchanged in hue but check contrast: `coral-pulse` and `leaf-bright` both pass WCAG AA against the dark background as-is; verify during implementation and lighten by ~8% if not
- Triage colors: keep hue, raise lightness slightly (e.g. `triage-red` → a lighter red than the light-mode value) so they stay legible on a dark surface without looking washed out

Font: DM Sans (fallback Inter, system-ui). Shapes: buttons `40px` radius, cards `24px`, tags `9999px`; shadow `rgba(0,0,0,0.05) 0px 1px 1px 0px` in light mode, `rgba(0,0,0,0.4) 0px 1px 2px 0px` in dark mode (shadows need more opacity to read on a dark surface). Animations: `slide-up` (chat message entry, 0.3s ease-out), `fade-in`, `blink` (loading/active states).

## 5. Dark Mode Toggle

A `ThemeToggle` component (sun/moon `lucide-react` icon that swaps on click) lives in the header on both the patient nav and the doctor portal header — this is a required, visible control, not just underlying theme support.

- **Default:** on first load, read `window.matchMedia('(prefers-color-scheme: dark)')` and apply that as the initial theme.
- **Override & persistence:** once the user taps the toggle, store the explicit choice in `localStorage['triageplus-theme']` (`"light"` | `"dark"`) and it takes precedence over the system preference on every subsequent load.
- **Application:** toggle the `dark` class on `<html>`; every color token above must have a `dark:` Tailwind variant wired up, not just the palette values existing unused in the config.
- Applies uniformly across patient routes and `/doctor/*` (the doctor palette's dark variant is defined in §7).

## 6. ChatWindow (patient side)

Manages the WebSocket connection to `/ws/chat/{session_id}` and renders incoming state updates:

- `{"type": "message"}` — standard chat bubble, `slide-up` animation.
- `{"type": "typing"}` — typing indicator, `blink` animation.
- `{"type": "emergency"}` — renders a non-dismissible, full-width red banner (or modal) above the chat, using `triage-red`, with the backend's guidance text and a clear call-to-action. Takes visual priority over everything else on screen and stays until the patient acknowledges it; don't let normal chat scrolling bury it.

**Opening quick-reply chips:** the first bot turn (right after intake) offers three tappable chips — "Describe my symptoms," "Book by department," "Book with a specific doctor." Tapping a chip sends its label text as a normal chat message; the backend's `node_detect_intent` handles it exactly like free-typed text, so this is a UI convenience only, not a separate code path.

**Department choice chips:** when the backend asks the patient to pick a department (`awaiting_department_choice` in the AI engine, see AI engine prompt §4.2), render the department list as tappable chips sourced from `GET /api/v1/specialties` — never hardcode the department names in the frontend, since they must always match what the backend can actually resolve. Tapping a chip sends that department's name as the reply.

**Doctor picker (optional but recommended for the "browse doctors" pattern):** on the landing page or as a chip flow, allow browsing `GET /api/v1/doctors?specialty_id=` for a chosen specialty, showing name, rating, and average consult time, with a "Book with this doctor" action that seeds the chat with "book with Dr. {name}."

**Mic button:** a `Mic` icon button (Lucide) sits in the chat input bar next to the text field. Ship it **disabled**, with `aria-label="Voice input — coming soon"` and a tooltip on hover. Wire an `onClick` stub (e.g., a toast: "Voice input is coming soon") so activating full voice input later is a one-line change, not a layout change.

**Booking flow UI:** inline slot-selection cards rendered from the backend's slot-offer message (works identically whether the offer came from symptom-triage or direct booking), a payment-simulation step, and a confirmation card. Confidence values arriving over the socket are 0–1 floats — format as a percentage (`Math.round(confidence * 100)}%`) only at render time, nowhere else; direct-booking confirmations have no confidence value at all and must render without a confidence line rather than showing "null%" or "0%".

## 7. Mobile Responsiveness (binding for every screen)

Build mobile-first: base Tailwind classes target the smallest viewport, `sm:`/`md:`/`lg:` classes layer on enhancements for larger screens — never the reverse.

**Breakpoints** (Tailwind defaults): `sm` 640px, `md` 768px, `lg` 1024px, `xl` 1280px. Test matrix for every screen: 375px (iPhone SE — smallest realistic target), 390px (iPhone 12/13), 768px (iPad portrait), 1024px+ (desktop).

**ChatWindow specifics:**
- Full-height layout uses `100dvh`, not `100vh` — mobile browser chrome (address bar) resizes `vh` and causes layout jumps; `dvh` accounts for this.
- Input bar is pinned to the bottom with `padding-bottom: env(safe-area-inset-bottom)` so it clears the home-indicator area on notched iOS devices.
- Message bubble max-width: 85% of viewport on mobile (`<sm`), 65% on desktop (`lg:max-w-[65%]`).
- No horizontal scroll under any circumstance — long unbroken strings (e.g. a long doctor name) must wrap, not overflow.
- On-screen keyboard handling: when the keyboard opens on mobile, the input bar and the most recent 1–2 messages must remain visible above it. Use the `visualViewport` API to detect the resize and adjust scroll position rather than relying on default browser behavior, which can hide the input behind the keyboard on some Android browsers.

**Doctor Portal specifics:**
- Sidebar collapses below `md`: either a bottom tab bar (Dashboard / Queue / Appointments / Patients) or a hamburger-triggered drawer — pick one and apply it consistently, don't mix patterns across pages.
- The Triage Queue `DataTable` becomes a stacked card list below `md` (one card per patient, key fields only: name, wait time, severity badge, chief complaint) rather than a horizontally-scrolling table — horizontal scroll on a data table is a common mobile-usability failure and should be avoided here specifically.
- KPI cards on the Dashboard stack to a single column below `sm`, two columns at `sm`–`md`, four columns at `lg+`.
- The `Sheet` quick-view slides up from the bottom (`side="bottom"`) below `md`, and from the right (`side="right"`) at `md` and above — shadcn's `Sheet` supports this via the `side` prop; don't ship the desktop right-side variant on mobile, where it eats too much horizontal space.
- Touch targets: every interactive element (buttons, badges used as filters, table-row taps) is at least 44×44px, per WCAG/Apple HIG guidance — this applies on both the patient and doctor sides.

## 8. Doctor Portal (`/doctor/*`)

Built on shadcn/ui + Radix UI primitives + Tailwind CSS + Recharts, using a "Clinical Blue & Slate" theme distinct from the patient-facing palette, with its own light and dark variants:

| Role | Light | Dark |
|---|---|---|
| Primary | `#2563EB` | `#3b82f6` (lightened for contrast on dark) |
| Background | `#FFFFFF` | `#020817` (slate-950) |
| Surface (cards) | `#FFFFFF` | `#0f172a` (slate-900) |
| Muted | `#F1F5F9` | `#1e293b` (slate-800) |
| Text | `#020817` | `#f1f5f9` (slate-100) |
| Text muted | `#64748B` | `#94a3b8` (slate-400) |
| Critical (Level 1–2) | `#EF4444` | `#f87171` (lightened) |
| Urgent (Level 3) | `#F59E0B` | `#fbbf24` (lightened) |
| Stable (Level 4–5) | `#16A34A` | `#4ade80` (lightened) |

All requests attach `Authorization: Bearer <token>` from `supabase.auth.getSession()`; a route guard redirects to `/doctor/login` on `401`.

### Layout
- **Sidebar (desktop) / bottom nav or drawer (mobile, see §7):** logo, nav links (Dashboard, Triage Queue, Appointments, Patient Directory, Settings), doctor profile + logout, theme toggle.
- **Header:** global `Command` palette search, notification bell, status toggle (`On Call` / `In Surgery` / `Available` / `Offline`).

### Dashboard — `GET /api/v1/doctor/dashboard`
KPI `Card`s: Total Patients Waiting, Critical Cases (`triage_level` 1–2, destructive styling), Upcoming Appointments Today, Average Wait Time. Below: a `Tabs`/`Card` summary of the next 3–5 patients by `triage_level`, and a `ScrollArea` activity stream.

### Triage Queue — `GET /api/v1/doctor/queue`
`DataTable` columns (desktop) / stacked cards (mobile, see §7): Patient Name, Wait Time, Severity, Chief Complaint, Vitals Summary, Actions.

```tsx
// Severity badge — triage_level is ESI convention: 1 = most critical
function severityVariant(triage_level: number) {
  if (triage_level <= 2) return "destructive";
  if (triage_level === 3) return "warning";
  return "secondary"; // 4-5
}
```

Sort by wait time, filter by `triage_level` via `DropdownMenu`. Clicking a row opens a `Sheet` with triage notes, vitals history, and a notes/prescription input — `PATCH /api/v1/doctor/appointments/{id}` on save, `DELETE /api/v1/doctor/appointments/{id}` for cancellation.

### Appointments — `GET /api/v1/doctor/appointments?date=`
`Calendar`/`DatePicker` plus an agenda list (`Avatar` + `Card` per slot), "Start Visit" / "Reschedule" actions.

### Patient Directory & Detail — `GET /api/v1/doctor/patients`, `GET /api/v1/doctor/patients/{id}`
Directory: search by name/contact. Detail: profile header with allergies (`Badge`, red) and blood type; `Accordion` for medical history; `Tabs` for Vitals & Notes (Recharts line charts), Labs & Imaging (`Table`), Treatment Plan (`react-hook-form` + `Select` + `Input`).

### Required shadcn/ui components
`Card`, `Sheet`, `Tabs`, `Table`/DataTable, `Badge`, `Avatar`, `Button`, `Input`, `Textarea`, `Select`, `Accordion`, `ScrollArea`, `Command`, `DropdownMenu`, `Calendar`. Install via `npx shadcn-ui@latest add <component>`. Icons: `Activity`, `AlertCircle`, `Calendar`, `Users`, `Search`, `Bell`, `Sun`, `Moon` from `lucide-react`. Charts: `recharts` `<LineChart>` for vitals trends.

## 9. RAG Diagnostic Monitor (`/diagnostics`)

`RagMonitor.tsx`: a "Developer Access" lock screen asks for `DEVELOPER_PASSWORD`, stores it in `sessionStorage('developer_password')`, passes it as `?token=` on the WebSocket connection to `wss://<host>/api/v1/ws/diagnostics`. Wrong token → backend closes with `1008` → frontend bounces back to the lock screen. This is an internal developer tool; keep the auth model to the shared token, and it's fine for this one route to be desktop-oriented only (no mobile layout work required here).

Each broadcast event renders as a card: node identifier (indigo), extracted `E_codes`, `final_diagnosis` (or "Pending…") with the confidence percentage (formatted at render time from the 0–1 float, blank/omitted for direct-booking sessions where confidence is null), a collapsible "Retrieved RAG Chunks (N)" accordion, and a latency footer when `latencies` is present.

## 10. Acceptance Tests

- An emergency WS message renders the non-dismissible red banner within one render cycle and it persists across subsequent normal messages until acknowledged.
- The mic button is visibly present, disabled, and accessible (screen reader announces "Voice input — coming soon").
- Confidence is never displayed anywhere without the `* 100` formatting step and never appears more than once per value; direct-booking confirmations render cleanly with no confidence line at all.
- Tapping each opening quick-reply chip produces the same backend `intent` result as typing the equivalent sentence manually.
- Department chips always match the live contents of `GET /api/v1/specialties` — changing a specialty name in the database changes the chip label without a frontend code change.
- Triage queue badges: test fixtures at `triage_level` 1, 3, and 5 render `destructive`, `warning`, and `secondary` respectively, in both light and dark mode.
- Theme toggle: reloading the page after toggling to dark mode preserves dark mode (via `localStorage`); clearing `localStorage` falls back to the OS-level `prefers-color-scheme`.
- At 375px width: no horizontal scroll anywhere in the patient chat or the Doctor Portal; the Triage Queue renders as stacked cards, not a scrolling table; the chat input bar stays visible and usable with the on-screen keyboard open.
- Any `/doctor/*` route redirects to `/doctor/login` on a `401` from its data fetch.
- Wrong password on `/diagnostics` bounces back to the lock screen and never opens the socket.
