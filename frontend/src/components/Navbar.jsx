import './Navbar.css'

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar__brand">
        <span className="navbar__logo">⬡</span>
        <span className="navbar__name">MemoryOS</span>
      </div>
      <div className="navbar__actions">
        <span className="navbar__badge">Beta</span>
      </div>
    </nav>
  )
}
