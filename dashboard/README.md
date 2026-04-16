# Threat Detector Dashboard

An editorial minimalist cybersecurity dashboard inspired by Zara's design philosophy.

## Design Philosophy

This dashboard transforms security monitoring from chaotic alerts into a composed intelligence briefing room.

### Core Principles

1. **Editorial Storytelling Over Transactional UI**
   - Feel quiet, controlled, and confident
   - Emphasize context and narrative ("What's happening right now?")
   - Reveal details progressively, not all at once

2. **Visual Language**
   - Grayscale + semantic accent colors only
   - Muted amber → anomaly detected
   - Deep red → confirmed threat
   - Cool blue → AI analysis/insight
   - Never use saturated colors as decoration

3. **Layout & Information Hierarchy**
   - One dominant visual per screen section
   - Full-width data visualizations as hero content
   - High whitespace, minimal dividers
   - Magazine-like sections, not dashboard widgets

4. **Typography**
   - Large, calm headings (no ALL CAPS alerts)
   - Numbers speak louder than labels
   - Quiet, restrained body text

5. **Navigation**
   - Minimal, collapsed by default
   - Icon-only sidebar expands on hover
   - Primary sections: Overview, Incidents, Models, Logs

6. **Progressive Disclosure**
   - Threat summary first
   - Click → explanation
   - Click again → raw data
   - Builds trust in the AI system

7. **CTAs & Actions**
   - Text-only buttons, never dominant
   - Appear only when relevant
   - Subtle underlines instead of colored backgrounds

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling with semantic color system
- **Recharts** - Data visualizations
- **Lucide React** - Icon library
- **React Router** - Navigation

## Project Structure

```
dashboard/
├── src/
│   ├── pages/
│   │   ├── Overview.tsx      # Hero editorial screen
│   │   ├── Incidents.tsx     # Incident details
│   │   ├── Models.tsx        # Model management
│   │   └── Logs.tsx          # Audit logs
│   ├── components/
│   │   ├── Layout.tsx        # Minimal navigation
│   │   ├── ThreatTimeline.tsx
│   │   ├── ModelConfidence.tsx
│   │   └── IncidentList.tsx
│   ├── styles/
│   │   └── globals.css       # Design system CSS
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── tailwind.config.js        # Semantic color palette
├── vite.config.ts
└── package.json
```

## Color System

- **Threat**: `#DC2626` (deep red)
- **Anomaly**: `#D97706` (muted amber)
- **Insight**: `#2563EB` (cool blue)
- **Neutral**: Black, white, grays

## Running the Dashboard

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The dashboard will be available at `http://localhost:3000`

## API Integration

The dashboard proxies API requests to `http://localhost:8000/api`

Update `vite.config.ts` proxy settings to match your backend URL.

## Design Inspiration

This dashboard follows Zara's editorial design principles:
- Minimal visual hierarchy
- High whitespace utilization
- Photography as hero content (replaced with data visualizations)
- Calm, authoritative typography
- Subtle interaction design
- Context-driven information revelation

The result: a security dashboard that feels like a premium intelligence briefing rather than a chaotic alert console.
