'use client';

import { createContext, useContext, useMemo, useState, useCallback } from 'react';
import { RoomContext } from '@livekit/components-react';
import { APP_CONFIG_DEFAULTS, type AppConfig } from '@/app-config';
import { useRoom } from '@/hooks/useRoom';

const SessionContext = createContext<{
  appConfig: AppConfig;
  isSessionActive: boolean;
  startSession: () => void;
  endSession: () => void;
  startSessionWithName: (name?: string) => void;
}>({
  appConfig: APP_CONFIG_DEFAULTS,
  isSessionActive: false,
  startSession: () => {},
  endSession: () => {},
  startSessionWithName: () => {},
});

interface SessionProviderProps {
  appConfig: AppConfig;
  children: React.ReactNode;
}

export const SessionProvider = ({ appConfig, children }: SessionProviderProps) => {
  const [currentAppConfig, setCurrentAppConfig] = useState<AppConfig>(appConfig);

  const { room, isSessionActive, startSession, endSession } = useRoom(currentAppConfig);

  const startSessionWithName = useCallback(
    (name?: string) => {
      if (name) setCurrentAppConfig((c) => ({ ...c, agentName: name }));
      // call startSession and pass the provided name so the token and the data publish
      // are created with the correct agent name immediately, avoiding a state race.
      startSession(name);
    },
    [startSession]
  );

  const contextValue = useMemo(
    () => ({ appConfig: currentAppConfig, isSessionActive, startSession, endSession, startSessionWithName }),
    [currentAppConfig, isSessionActive, startSession, endSession, startSessionWithName]
  );

  return (
    <RoomContext.Provider value={room}>
      <SessionContext.Provider value={contextValue}>{children}</SessionContext.Provider>
    </RoomContext.Provider>
  );
};

export function useSession() {
  return useContext(SessionContext);
}
