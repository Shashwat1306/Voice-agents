import { useEffect, useState, useRef } from 'react';

type LogEntry = {
  timestamp: string;
  role: 'gm' | 'player' | string;
  text: string;
};

export function useGameplayLog(pollInterval = 1500) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const etagRef = useRef<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function fetchOnce() {
      try {
        const res = await fetch('/api/gameplay-log');
        if (!res.ok) return;
        const payload = await res.json();
        const data: LogEntry[] = Array.isArray(payload?.data) ? payload.data : payload?.data || [];
        if (!mounted) return;
        setEntries(data);
      } catch (e) {
        // ignore network errors
      }
    }

    fetchOnce();
    const id = setInterval(fetchOnce, pollInterval);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [pollInterval]);

  return entries;
}
