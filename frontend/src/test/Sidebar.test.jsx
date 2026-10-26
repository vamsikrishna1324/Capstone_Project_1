import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Sidebar from '../components/Sidebar';

describe('Sidebar Component', () => {
  const mockHealth = {
    status: 'ok',
    api: 'online',
    database: { connected: true, host: '127.0.0.1' },
    edge_impulse: { configured: true, status: 'configured' },
  };

  it('renders brand identity and all 5 navigation tabs', () => {
    render(<Sidebar activeTab="dashboard" onTabChange={vi.fn()} health={mockHealth} />);

    // Brand title
    expect(screen.getByText('EdgeBench')).toBeInTheDocument();
    expect(screen.getByText('Capstone Studio')).toBeInTheDocument();

    // 5 Navigation Tabs
    expect(screen.getByRole('button', { name: /Overview/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Model Comparison/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run Benchmark/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Recommendations/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run History/i })).toBeInTheDocument();
  });

  it('highlights the active tab and switches tab on click', () => {
    const handleTabChange = vi.fn();
    render(<Sidebar activeTab="dashboard" onTabChange={handleTabChange} health={mockHealth} />);

    const dashboardBtn = screen.getByRole('button', { name: /Overview/i });
    expect(dashboardBtn.className).toContain('active');

    const compareBtn = screen.getByRole('button', { name: /Model Comparison/i });
    expect(compareBtn.className).not.toContain('active');

    fireEvent.click(compareBtn);
    expect(handleTabChange).toHaveBeenCalledWith('compare');
  });

  it('renders verified runtime telemetry when connected', () => {
    render(<Sidebar activeTab="dashboard" onTabChange={vi.fn()} health={mockHealth} />);

    expect(screen.getByText('FastAPI')).toBeInTheDocument();
    expect(screen.getByText('Online')).toBeInTheDocument();

    expect(screen.getByText('MySQL')).toBeInTheDocument();
    expect(screen.getByText('Connected')).toBeInTheDocument();

    expect(screen.getByText('Edge Impulse')).toBeInTheDocument();
    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('displays honest unknown/checking state when health is null', () => {
    render(<Sidebar activeTab="dashboard" onTabChange={vi.fn()} health={null} />);

    const checkingBadges = screen.getAllByText('Checking...');
    expect(checkingBadges.length).toBeGreaterThanOrEqual(2);
  });

  it('displays degraded/offline state when services are disconnected', () => {
    const degradedHealth = {
      status: 'error',
      api: 'unreachable',
      database: { connected: false },
      edge_impulse: { configured: false },
    };

    render(<Sidebar activeTab="dashboard" onTabChange={vi.fn()} health={degradedHealth} />);

    expect(screen.getByText('Degraded')).toBeInTheDocument();
    expect(screen.getByText('Offline')).toBeInTheDocument();
    expect(screen.getByText('Not Configured')).toBeInTheDocument();
  });

  it('triggers onToggleCollapse when collapse button is clicked', () => {
    const handleToggle = vi.fn();
    render(
      <Sidebar
        activeTab="dashboard"
        onTabChange={vi.fn()}
        health={mockHealth}
        isCollapsed={false}
        onToggleCollapse={handleToggle}
      />
    );

    const toggleBtn = screen.getByRole('button', { name: /Collapse sidebar/i });
    expect(toggleBtn).toBeInTheDocument();
    fireEvent.click(toggleBtn);
    expect(handleToggle).toHaveBeenCalledTimes(1);
  });

  it('renders collapsed icon rail with accessible tooltips and preserves active tab highlighting', () => {
    render(
      <Sidebar
        activeTab="compare"
        onTabChange={vi.fn()}
        health={mockHealth}
        isCollapsed={true}
        onToggleCollapse={vi.fn()}
      />
    );

    // Sidebar element has collapsed class
    const sidebar = document.getElementById('app-sidebar');
    expect(sidebar.className).toContain('collapsed');

    // Branding text is hidden
    expect(screen.queryByText('EdgeBench')).not.toBeInTheDocument();
    expect(screen.queryByText('Capstone Studio')).not.toBeInTheDocument();

    // Toggle button now shows Expand tooltip
    expect(screen.getByRole('button', { name: /Expand sidebar/i })).toBeInTheDocument();

    // Nav items have accessible labels and titles
    const compareBtn = screen.getByRole('button', { name: /Model Comparison/i });
    expect(compareBtn).toBeInTheDocument();
    expect(compareBtn.getAttribute('title')).toBe('Model Comparison');
    expect(compareBtn.className).toContain('active');

    const overviewBtn = screen.getByRole('button', { name: /Overview/i });
    expect(overviewBtn.getAttribute('title')).toBe('Overview');
    expect(overviewBtn.className).not.toContain('active');

    // Telemetry indicators remain accessible via tooltips
    const telemetryContainer = document.getElementById('sidebar-telemetry');
    expect(telemetryContainer).toBeInTheDocument();
  });
});

