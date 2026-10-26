import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import DashboardHeader from '../components/DashboardHeader';

describe('DashboardHeader Component', () => {
  it('renders overview title and subtitle for dashboard tab', () => {
    render(<DashboardHeader activeTab="dashboard" onRefresh={vi.fn()} refreshing={false} />);

    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(
      screen.getByText('Automated image model benchmarking and selection for edge deployment.')
    ).toBeInTheDocument();
  });

  it('renders section title dynamically for other tabs', () => {
    const { rerender } = render(<DashboardHeader activeTab="compare" onRefresh={vi.fn()} />);
    expect(screen.getByText('Model Comparison Matrix')).toBeInTheDocument();

    rerender(<DashboardHeader activeTab="benchmark" onRefresh={vi.fn()} />);
    expect(screen.getByText('Run Evaluation & Profiling')).toBeInTheDocument();

    rerender(<DashboardHeader activeTab="recommendations" onRefresh={vi.fn()} />);
    expect(screen.getByText('Deployment Recommendations')).toBeInTheDocument();

    rerender(<DashboardHeader activeTab="history" onRefresh={vi.fn()} />);
    expect(screen.getByText('Run History & Sessions')).toBeInTheDocument();
  });

  it('triggers onRefresh when clicking Sync Repository button', () => {
    const handleRefresh = vi.fn();
    render(<DashboardHeader activeTab="dashboard" onRefresh={handleRefresh} refreshing={false} />);

    const refreshBtn = screen.getByRole('button', { name: /Sync Repository/i });
    fireEvent.click(refreshBtn);
    expect(handleRefresh).toHaveBeenCalledTimes(1);
  });

  it('shows syncing state and disables button when refreshing is true', () => {
    render(<DashboardHeader activeTab="dashboard" onRefresh={vi.fn()} refreshing={true} />);

    const syncingBtn = screen.getByRole('button', { name: /Syncing.../i });
    expect(syncingBtn).toBeDisabled();
  });
});
