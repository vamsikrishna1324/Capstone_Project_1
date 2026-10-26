import React from 'react';
import {
  LayoutDashboard,
  Layers,
  PlayCircle,
  Award,
  History,
  Cpu,
  Database,
  X,
  Zap,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  onTabChange,
  health,
  isOpen = false,
  onClose,
  isCollapsed = false,
  onToggleCollapse,
}) {
  const isHealthy = health?.status === 'ok' || health?.api === 'online';
  const isDbConnected = health?.database?.connected === true;
  const isEdgeImpulseReady = health?.edge_impulse?.configured === true;

  const navItems = [
    { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
    { id: 'compare', label: 'Model Comparison', icon: Layers },
    { id: 'benchmark', label: 'Run Benchmark', icon: PlayCircle },
    { id: 'recommendations', label: 'Recommendations', icon: Award },
    { id: 'history', label: 'Run History', icon: History },
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          id="sidebar-backdrop"
          onClick={onClose}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.65)',
            backdropFilter: 'blur(4px)',
            zIndex: 90,
          }}
        />
      )}

      <aside
        id="app-sidebar"
        className={`app-sidebar ${isOpen ? 'open' : ''} ${isCollapsed ? 'collapsed' : ''}`}
        aria-label="Application Sidebar"
      >
        {/* Brand / Logo Section & Collapse Toggle */}
        <div
          style={{
            padding: isCollapsed ? '1rem 0.5rem' : '1.25rem 1.25rem 1rem',
            borderBottom: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            justifyContent: isCollapsed ? 'center' : 'space-between',
            flexDirection: isCollapsed ? 'column' : 'row',
            gap: isCollapsed ? '0.75rem' : '0',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.65rem',
              justifyContent: isCollapsed ? 'center' : 'flex-start',
            }}
          >
            <div
              title="EdgeBench Capstone Studio"
              style={{
                width: '34px',
                height: '34px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(37, 99, 235, 0.4)',
                flexShrink: 0,
              }}
            >
              <Cpu size={19} color="#ffffff" />
            </div>
            {!isCollapsed && (
              <div>
                <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
                  EdgeBench
                </div>
                <div style={{ fontSize: '0.675rem', fontWeight: 600, color: '#38bdf8', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                  Capstone Studio
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            {/* Desktop Collapse / Expand Toggle Button */}
            {onToggleCollapse && (
              <button
                id="btn-toggle-sidebar-collapse"
                onClick={onToggleCollapse}
                aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                className="sidebar-collapse-btn"
                style={{
                  background: 'transparent',
                  border: '1px solid #334155',
                  borderRadius: '6px',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  padding: '0.35rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 150ms ease',
                }}
              >
                {isCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
              </button>
            )}

            {/* Close button on mobile */}
            {onClose && (
              <button
                onClick={onClose}
                className="sidebar-close-btn"
                aria-label="Close sidebar"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  display: 'none',
                  padding: '0.25rem',
                }}
              >
                <X size={18} />
              </button>
            )}
          </div>
        </div>

        {/* Navigation Section */}
        <nav
          aria-label="Main Navigation"
          style={{
            padding: isCollapsed ? '1rem 0.4rem' : '1rem 0.75rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
            flex: 1,
            overflowY: 'auto',
          }}
        >
          {!isCollapsed && (
            <div
              style={{
                fontSize: '0.675rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                color: '#64748b',
                padding: '0.25rem 0.65rem 0.5rem',
              }}
            >
              Workspace
            </div>
          )}

          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                id={`tab-${item.id}`}
                title={item.label}
                aria-label={item.label}
                onClick={() => {
                  onTabChange(item.id);
                  if (onClose) onClose();
                }}
                className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: isCollapsed ? 'center' : 'flex-start',
                  gap: isCollapsed ? '0' : '0.75rem',
                  width: '100%',
                  padding: isCollapsed ? '0.65rem 0' : '0.625rem 0.75rem',
                  borderRadius: '6px',
                  border: 'none',
                  borderLeft: isActive ? '3px solid #3b82f6' : '3px solid transparent',
                  background: isActive ? 'rgba(37, 99, 235, 0.16)' : 'transparent',
                  color: isActive ? '#ffffff' : '#94a3b8',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 600 : 500,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 150ms ease',
                }}
              >
                <Icon
                  size={18}
                  color={isActive ? '#60a5fa' : '#64748b'}
                  style={{ flexShrink: 0 }}
                />
                {!isCollapsed && <span style={{ flex: 1 }}>{item.label}</span>}
              </button>
            );
          })}
        </nav>

        {/* System Health / Telemetry Status (Footer) */}
        <div
          id="sidebar-telemetry"
          style={{
            padding: isCollapsed ? '0.85rem 0.4rem' : '1rem 0.85rem',
            borderTop: '1px solid #1e293b',
            background: 'rgba(11, 17, 32, 0.6)',
          }}
        >
          {!isCollapsed && (
            <div
              style={{
                fontSize: '0.675rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                color: '#64748b',
                marginBottom: '0.6rem',
                paddingLeft: '0.25rem',
              }}
            >
              Runtime Telemetry
            </div>
          )}

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: isCollapsed ? 'center' : 'stretch',
              gap: '0.45rem',
            }}
          >
            {/* FastAPI Backend */}
            <div
              id="telemetry-fastapi"
              title={`FastAPI: ${health ? (isHealthy ? 'Online' : 'Degraded') : 'Checking...'}`}
              aria-label={`FastAPI: ${health ? (isHealthy ? 'Online' : 'Degraded') : 'Checking...'}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: isCollapsed ? 'center' : 'space-between',
                padding: isCollapsed ? '0.45rem' : '0.35rem 0.55rem',
                borderRadius: '5px',
                background: '#1e293b',
                fontSize: '0.75rem',
                width: isCollapsed ? '36px' : 'auto',
                height: isCollapsed ? '36px' : 'auto',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: isHealthy ? '#10b981' : '#f59e0b',
                    boxShadow: isHealthy ? '0 0 6px #10b981' : 'none',
                    display: 'inline-block',
                    flexShrink: 0,
                  }}
                />
                {!isCollapsed && <span style={{ color: '#cbd5e1' }}>FastAPI</span>}
              </div>
              {!isCollapsed && (
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    color: isHealthy ? '#6ee7b7' : '#fcd34d',
                  }}
                >
                  {health ? (isHealthy ? 'Online' : 'Degraded') : 'Checking...'}
                </span>
              )}
            </div>

            {/* MySQL Database */}
            <div
              id="telemetry-mysql"
              title={`MySQL: ${health ? (isDbConnected ? 'Connected' : 'Offline') : 'Checking...'}`}
              aria-label={`MySQL: ${health ? (isDbConnected ? 'Connected' : 'Offline') : 'Checking...'}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: isCollapsed ? 'center' : 'space-between',
                padding: isCollapsed ? '0.45rem' : '0.35rem 0.55rem',
                borderRadius: '5px',
                background: '#1e293b',
                fontSize: '0.75rem',
                width: isCollapsed ? '36px' : 'auto',
                height: isCollapsed ? '36px' : 'auto',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Database size={14} color={isDbConnected ? '#10b981' : '#ef4444'} style={{ flexShrink: 0 }} />
                {!isCollapsed && <span style={{ color: '#cbd5e1' }}>MySQL</span>}
              </div>
              {!isCollapsed && (
                <span
                  style={{
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    color: isDbConnected ? '#6ee7b7' : '#fca5a5',
                  }}
                >
                  {health ? (isDbConnected ? 'Connected' : 'Offline') : 'Checking...'}
                </span>
              )}
            </div>

            {/* Edge Impulse Integration */}
            {health?.edge_impulse && (
              <div
                id="telemetry-edge-impulse"
                title={`Edge Impulse: ${isEdgeImpulseReady ? 'Ready' : 'Not Configured'}`}
                aria-label={`Edge Impulse: ${isEdgeImpulseReady ? 'Ready' : 'Not Configured'}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: isCollapsed ? 'center' : 'space-between',
                  padding: isCollapsed ? '0.45rem' : '0.35rem 0.55rem',
                  borderRadius: '5px',
                  background: '#1e293b',
                  fontSize: '0.75rem',
                  width: isCollapsed ? '36px' : 'auto',
                  height: isCollapsed ? '36px' : 'auto',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                  <Zap size={14} color={isEdgeImpulseReady ? '#38bdf8' : '#94a3b8'} style={{ flexShrink: 0 }} />
                  {!isCollapsed && <span style={{ color: '#cbd5e1' }}>Edge Impulse</span>}
                </div>
                {!isCollapsed && (
                  <span
                    style={{
                      fontWeight: 600,
                      fontSize: '0.7rem',
                      color: isEdgeImpulseReady ? '#7dd3fc' : '#94a3b8',
                    }}
                  >
                    {isEdgeImpulseReady ? 'Ready' : 'Not Configured'}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
