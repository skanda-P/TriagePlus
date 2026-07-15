# Design System & Components - Doctor Portal (Vite + React)

This document outlines the design language, color palette, and UI components required to build the Medical Triage System's Doctor Portal. All UI components are based on **shadcn/ui** (which utilizes Radix UI primitives and Tailwind CSS).

## 1. Color Palette

The portal uses a "Clinical Blue & Slate" theme to convey trust, cleanliness, and urgency where necessary.

| Color Role | Tailwind Class / HSL Value | Hex Code | Usage Context |
| :--- | :--- | :--- | :--- |
| **Primary** | `bg-primary` / `hsl(221.2 83.2% 53.3%)` | `#2563EB` (Blue 600) | Main buttons, active tabs, primary navigation. |
| **Background** | `bg-background` / `hsl(0 0% 100%)` | `#FFFFFF` (White) | Main application background. |
| **Surface (Card)** | `bg-card` / `hsl(0 0% 100%)` | `#FFFFFF` (White) | Background for cards, sheets, and dropdowns. |
| **Muted/Secondary** | `bg-muted` / `hsl(210 40% 96.1%)` | `#F1F5F9` (Slate 100) | Secondary buttons, table headers, hover states. |
| **Text Primary** | `text-foreground` / `hsl(222.2 84% 4.9%)` | `#020817` (Slate 950)| Standard body text, headings. |
| **Text Muted** | `text-muted-foreground` / `hsl(215.4 16.3% 46.9%)` | `#64748B` (Slate 500)| Subtitles, secondary text, placeholders. |
| **Critical / Level 1** | `bg-destructive` / `hsl(0 84.2% 60.2%)` | `#EF4444` (Red 500) | Resuscitation/Emergent triage badges, critical alerts. |
| **Urgent / Level 3** | `bg-amber-500` / `hsl(38 92% 50%)` | `#F59E0B` (Amber 500)| Urgent triage badges, warnings. |
| **Stable / Level 5** | `bg-green-600` / `hsl(142.1 76.2% 36.3%)` | `#16A34A` (Green 600)| Non-urgent triage badges, successful actions. |

---

## 2. Shadcn UI Components

Below is the list of required `shadcn/ui` components. To use them in your Vite project, you must first initialize shadcn (`npx shadcn-ui@latest init`) and then run the respective add commands.

### Layout & Containers
* **Card** 
    * *Usage:* KPI dashboards, appointment slots.
    * *Command:* `npx shadcn-ui@latest add card`
    * *Link:* [shadcn - Card](https://ui.shadcn.com/docs/components/card)
* **Sheet (Slide-out panel)**
    * *Usage:* Quick-view for patient triage details when clicking a table row.
    * *Command:* `npx shadcn-ui@latest add sheet`
    * *Link:* [shadcn - Sheet](https://ui.shadcn.com/docs/components/sheet)
* **Tabs**
    * *Usage:* Switching between Vitals, Labs, and Treatment Plan in the patient view.
    * *Command:* `npx shadcn-ui@latest add tabs`
    * *Link:* [shadcn - Tabs](https://ui.shadcn.com/docs/components/tabs)

### Data Display
* **Data Table (Table)**
    * *Usage:* The main Triage Queue list.
    * *Command:* `npx shadcn-ui@latest add table`
    * *Link:* [shadcn - Data Table](https://ui.shadcn.com/docs/components/data-table)
* **Badge**
    * *Usage:* Triage severity levels (Red, Amber, Green).
    * *Command:* `npx shadcn-ui@latest add badge`
    * *Link:* [shadcn - Badge](https://ui.shadcn.com/docs/components/badge)
* **Avatar**
    * *Usage:* Doctor profile picture in the sidebar, patient icons.
    * *Command:* `npx shadcn-ui@latest add avatar`
    * *Link:* [shadcn - Avatar](https://ui.shadcn.com/docs/components/avatar)

### Forms & Inputs
* **Button**
    * *Usage:* "Start Visit", "Save Notes", "Acknowledge" actions.
    * *Command:* `npx shadcn-ui@latest add button`
    * *Link:* [shadcn - Button](https://ui.shadcn.com/docs/components/button)
* **Input & Textarea**
    * *Usage:* Search bar, doctor's triage notes.
    * *Command:* `npx shadcn-ui@latest add input textarea`
    * *Link:* [shadcn - Input](https://ui.shadcn.com/docs/components/input)
* **Select (Dropdown)**
    * *Usage:* Changing doctor status (On Call, Offline) or filtering the queue.
    * *Command:* `npx shadcn-ui@latest add select`
    * *Link:* [shadcn - Select](https://ui.shadcn.com/docs/components/select)

---

## 3. External Dependencies

### Iconography: Lucide React
Shadcn uses Lucide icons by default. They are clean and consistent.
* **Installation:** `npm install lucide-react`
* **Icons to use:** `Activity` (vitals), `AlertCircle` (critical severity), `Calendar` (appointments), `Users` (patient directory), `Search` (global search).

### Data Visualization: Recharts
Used for displaying simple trend lines (like heart rate or blood pressure over time) in the patient details sheet.
* **Installation:** `npm install recharts`
* **Component to use:** `<LineChart>` for vitals trends.

---

## 4. Example Component Composition (Triage Queue Row)

```jsx
import { TableRow, TableCell } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Activity } from "lucide-react"

// Example of how to use the elements together
export function TriageRow({ patient }) {
  return (
    <TableRow className="hover:bg-muted/50 cursor-pointer">
      <TableCell className="font-medium">{patient.name}</TableCell>
      <TableCell>{patient.waitTime} mins</TableCell>
      <TableCell>
        {/* Severity Badge using our defined colors */}
        <Badge "default"} "destructive" 1 : ? variant="{patient.severity">
          Level {patient.severity}
        </Badge>
      </TableCell>
      <TableCell className="text-muted-foreground">{patient.complaint}</TableCell>
      <TableCell>
        <Button size="sm" variant="outline">
          <Activity className="w-4 h-4 mr-2"/>
          View Vitals
        </Button>
      </TableCell>
    </TableRow>
  )
}