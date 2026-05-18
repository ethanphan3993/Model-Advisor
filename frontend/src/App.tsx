import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Brain, Scan, BookOpen, Database } from 'lucide-react'
import Home from './pages/Home'
// Wizard is the legacy guided flow; Home now does everything inline. We keep
// the route for back-compat with shareable links.
import Wizard from './pages/Wizard'
import Results from './pages/Results'
import ScanResults from './pages/ScanResults'
import Browse from './pages/Browse'
import ModelDetailPage from './pages/ModelDetail'
import Compare from './pages/Compare'
import Sources from './pages/Sources'

const navItems = [
  { to: '/', label: 'Home', icon: Brain },
  { to: '/scan', label: 'Hardware', icon: Scan },
  { to: '/browse', label: 'Browse', icon: BookOpen },
  { to: '/sources', label: 'Sources', icon: Database },
]

export default function App() {
  const { pathname } = useLocation()
  return (
    <div className="min-h-screen">
      <header className="border-b border-border/50 bg-background/95 backdrop-blur sticky top-0 z-40">
        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4">
          <Link to="/" className="flex items-center gap-2 text-lg font-semibold">
            <Brain className="h-6 w-6 text-primary" />
            <span>Model Advisor</span>
          </Link>
          <nav className="flex items-center gap-1">
            {navItems.map((item) => {
              const active = pathname === item.to || (item.to !== '/' && pathname.startsWith(item.to))
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors ${
                    active ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
                  }`}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              )
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/wizard/:useCaseId" element={<Wizard />} />
          <Route path="/results" element={<Results />} />
          <Route path="/scan" element={<ScanResults />} />
          <Route path="/browse" element={<Browse />} />
          <Route path="/model/:canonicalId" element={<ModelDetailPage />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/sources" element={<Sources />} />
        </Routes>
      </main>

      <footer className="border-t border-border/50 py-6 text-center text-xs text-muted-foreground">
        <div className="mx-auto max-w-5xl px-4">
          Model Advisor — public benchmark sources, hardware-aware ranking, agent-harness filter. macOS only for hardware scan.
        </div>
      </footer>
    </div>
  )
}
