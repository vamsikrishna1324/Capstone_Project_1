import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

export default function AlertModal({ isOpen, title, children, onConfirm, onCancel, confirmText = 'Proceed', cancelText = 'Cancel' }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '480px',
          background: 'var(--bg-surface-elevated)',
          border: '1px solid var(--border-medium)',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.6)',
          padding: '1.75rem',
          position: 'relative',
        }}
      >
        <button
          onClick={onCancel}
          style={{
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
          }}
          aria-label="Close dialog"
        >
          <X size={18} />
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '10px',
              background: 'rgba(245, 158, 11, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px var(--warning-glow)',
            }}
          >
            <AlertTriangle size={22} color="var(--warning)" />
          </div>
          <h3 id="modal-title" style={{ fontSize: '1.15rem' }}>
            {title}
          </h3>
        </div>

        <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6', marginBottom: '1.5rem' }}>
          {children}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <button id="modal-btn-cancel" onClick={onCancel} className="btn btn-secondary btn-sm">
            {cancelText}
          </button>
          <button id="modal-btn-confirm" onClick={onConfirm} className="btn btn-primary btn-sm" style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)' }}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
