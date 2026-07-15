# Comprehensive Prompt for Generating a Doctor Portal (Medical Triage System)

**Context:**
You are an expert Frontend Developer specializing in React, Vite, Tailwind CSS, and shadcn/ui. Your task is to build a modern, highly responsive, and accessible "Doctor Portal" for a Medical Triage System. The portal's main goal is to help doctors efficiently manage incoming patients based on severity, view their scheduled appointments, and access vital patient data quickly.

**Design System & Tech Stack:**
* **Framework:** React powered by Vite
* **Routing:** `react-router-dom` (essential for SPA navigation)
* **Styling:** Tailwind CSS
* **Component Library:** shadcn/ui, Radix UI primitives, Lucide React (icons), Recharts (for analytics).
* **Theme:** Clean, clinical, and high-contrast (white/slate backgrounds with semantic colors for medical urgency: Red for critical, Yellow/Orange for urgent, Green for stable).

---

### Core Layout Structure

1.  **Sidebar (shadcn `Sidebar` or `NavigationMenu`):**
    * Logo/Brand (e.g., "TriagePro - Doctor Portal").
    * Navigation Links (using `react-router-dom` `<Link>`): Dashboard, Triage Queue, Appointments, Patient Directory, Messages, Settings.
    * Bottom Section: Doctor Profile (`Avatar`, Name, Specialty), Log out button.
2.  **Top Header (App Bar):**
    * Global Search (`Command` palette): Quick search for patients by Name, ID, or Phone.
    * Notifications (`Bell` icon with a red notification badge for critical updates).
    * Current Status Toggle (`Select` or `DropdownMenu`): On Call, In Surgery, Available, Offline.

---

### Key Pages & Required Shadcn Components

#### 1. Dashboard (Overview)
* **KPI Cards (shadcn `Card`):** Display immediate metrics at the top.
    * "Total Patients Waiting"
    * "Critical Cases (Level 1 & 2)" - styled with destructive/red accents.
    * "Upcoming Appointments Today"
    * "Average Wait Time"
* **Quick Triage Overview (shadcn `Tabs` or `Card`):** A summary list of the next 3-5 patients needing immediate attention based on triage score.
* **Recent Activity Stream (shadcn `ScrollArea`):** Real-time updates on patient statuses (e.g., "Nurse updated vitals for John Doe", "New lab results available").

#### 2. Triage Queue (The Core Feature)
* **Interactive List (shadcn `DataTable`):** A robust data table to manage incoming triage.
    * **Columns:** Patient Name, Triage Wait Time, **Severity Level**, Chief Complaint, Vitals Summary, Actions.
    * **Severity Badges (shadcn `Badge`):** * `destructive` (Red/Black) for Level 1 (Resuscitation) / Level 2 (Emergent).
        * `warning` (Yellow/Orange) for Level 3 (Urgent).
        * `secondary` or `outline` (Blue/Gray/Green) for Level 4 (Less Urgent) / Level 5 (Non-Urgent).
    * **Sorting & Filtering:** Allow sorting by Wait Time and filtering by Severity using `DropdownMenu`.
* **Quick View (shadcn `Sheet`):** Clicking a table row should slide out a right-side sheet (`Sheet`) showing the patient's triage notes, vitals history, and an input area (`Textarea`, `Button`) to add doctor's notes, prescribe medication, or change triage status.

#### 3. Appointments Management
* **Calendar View (shadcn `Calendar` & `DatePicker`):**
    * A monthly/weekly view to visualize scheduled appointments.
* **Agenda List:** Next to the calendar, a vertical list of the day's appointments. Use `Avatar` for patient photos/initials, and `Card` for each appointment slot. Include buttons to "Start Visit" or "Reschedule".

#### 4. Patient Detail View
* **Profile Header:** Patient demographics, critical allergies (`Badge` in red), and blood type.
* **Medical History (shadcn `Accordion`):** Collapsible sections for Past Illnesses, Surgeries, Family History, and Chronic Conditions.
* **Current Triage Episode (shadcn `Tabs`):**
    * **Tab 1: Vitals & Notes** (Heart rate, BP, SpO2 with mini line-charts using Recharts).
    * **Tab 2: Labs & Imaging** (Lists of uploaded files/results using `Table`).
    * **Tab 3: Treatment Plan** (Form using `react-hook-form`, `Select`, and `Input` to order tests or prescribe).

---

### Specific Instructions for the AI Code Generation Model:

1.  **Code Quality & Modularity:** Break down the UI into logical components (e.g., `SidebarNavigation`, `StatCard`, `TriageDataTable`, `PatientSideSheet`). Create a standard Vite + React project structure.
2.  **Routing Setup:** Provide a basic setup using `react-router-dom` (`<BrowserRouter>`, `<Routes>`, `<Route>`) to handle navigation between the Dashboard and Triage Queue inside a persistent layout wrapper.
3.  **Mock Data:** Generate highly realistic medical mock data. Include realistic chief complaints, realistic vitals (BP: 140/90, HR: 110), and appropriate triage categorizations.
4.  **Interactivity:** Implement functional React state (`useState`) to allow toggling between Tabs, opening/closing the `Sheet` for patient details, and basic filtering on the `DataTable`.
5.  **Styling & Polish:** Use standard tailwind spacing (`p-4`, `gap-4`, `gap-6`). Utilize `text-muted-foreground` for secondary text. Ensure hover states (e.g., `hover:bg-muted` on table rows) are present for a tactile feel. Use Lucide-react icons for all UI iconography (e.g., Activity, Calendar, Users, AlertCircle).

**Final Prompt Execution Command:**
"Generate the complete React (Vite) code for this Doctor Portal. Ensure `react-router-dom` is used for client-side routing. Start with the main `App.jsx` router setup and Layout wrapper, followed by the Dashboard view and the Triage Queue component. Output the code block containing the main page structures, necessary Shadcn UI imports, and the mock data payload."
