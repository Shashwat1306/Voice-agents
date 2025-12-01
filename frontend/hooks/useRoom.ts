import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Room, RoomEvent, TokenSource } from 'livekit-client';
import { AppConfig } from '@/app-config';
import { toastAlert } from '@/components/livekit/alert-toast';

export function useRoom(appConfig: AppConfig) {
  const aborted = useRef(false);
  const room = useMemo(() => new Room(), []);
  const [isSessionActive, setIsSessionActive] = useState(false);

  useEffect(() => {
    function onDisconnected() {
      setIsSessionActive(false);
    }

    function onMediaDevicesError(error: Error) {
      toastAlert({
        title: 'Encountered an error with your media devices',
        description: `${error.name}: ${error.message}`,
      });
    }

    room.on(RoomEvent.Disconnected, onDisconnected);
    room.on(RoomEvent.MediaDevicesError, onMediaDevicesError);

    return () => {
      room.off(RoomEvent.Disconnected, onDisconnected);
      room.off(RoomEvent.MediaDevicesError, onMediaDevicesError);
    };
  }, [room]);

  useEffect(() => {
    return () => {
      aborted.current = true;
      room.disconnect();
    };
  }, [room]);

  const tokenSource = useMemo(
    () =>
      TokenSource.custom(async () => {
        const url = new URL(
          process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ?? '/api/connection-details',
          window.location.origin
        );

        try {
          const res = await fetch(url.toString(), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Sandbox-Id': appConfig.sandboxId ?? '',
            },
            body: JSON.stringify({
              room_config: appConfig.agentName
                ? {
                    agents: [{ agent_name: appConfig.agentName }],
                  }
                : undefined,
            }),
          });
          return await res.json();
        } catch (error) {
          console.error('Error fetching connection details:', error);
          throw new Error('Error fetching connection details!');
        }
      }),
    [appConfig]
  );

  const startSession = useCallback((overrideName?: string) => {
    setIsSessionActive(true);

    if (room.state === 'disconnected') {
      const { isPreConnectBufferEnabled } = appConfig;
      Promise.all([
        room.localParticipant.setMicrophoneEnabled(true, undefined, {
          preConnectBuffer: isPreConnectBufferEnabled,
        }),
        tokenSource
          .fetch({ agentName: overrideName ?? appConfig.agentName })
          .then((connectionDetails) =>
            room.connect(connectionDetails.serverUrl, connectionDetails.participantToken).then(() => {
              // publish a startup data message to notify any agent worker in the room
              try {
                const payload = JSON.stringify({ type: 'improv_start', player_name: overrideName ?? appConfig.agentName });
                const topic = 'improv';
                // debug: log the payload and topic to the browser console so we can verify what was sent
                // eslint-disable-next-line no-console
                console.debug('Publishing improv_start', { payload, topic });
                // publish as a Uint8Array to guarantee bytes arrive on all runtimes
                try {
                  const encoder = new TextEncoder();
                  const payloadBytes = encoder.encode(payload);
                  room.localParticipant.publishData(payloadBytes, { topic, reliable: true });
                } catch (e) {
                  // fallback: publish string if encoding not available for any reason
                  // eslint-disable-next-line no-console
                  console.warn('TextEncoder failed, publishing string payload instead', e);
                  // eslint-disable-next-line no-console
                  console.debug('Publishing improv_start (string fallback)', { payload, topic });
                  room.localParticipant.publishData(payload, { topic, reliable: true });
                }
              } catch (err) {
                // non-fatal: log and continue
                // eslint-disable-next-line no-console
                console.warn('Failed to publish improv_start message', err);
              }
            })
          ),
      ]).catch((error) => {
        if (aborted.current) {
          // Once the effect has cleaned up after itself, drop any errors
          //
          // These errors are likely caused by this effect rerunning rapidly,
          // resulting in a previous run `disconnect` running in parallel with
          // a current run `connect`
          return;
        }

        toastAlert({
          title: 'There was an error connecting to the agent',
          description: `${error.name}: ${error.message}`,
        });
      });
    }
  }, [room, appConfig, tokenSource]);

  const endSession = useCallback(() => {
    // publish an end-session signal to the room so the agent can clear state
    try {
      const payload = JSON.stringify({ type: 'improv_end' });
      const topic = 'improv';
      // eslint-disable-next-line no-console
      console.debug('Publishing improv_end', { payload, topic });
      try {
        const encoder = new TextEncoder();
        const payloadBytes = encoder.encode(payload);
        room.localParticipant.publishData(payloadBytes, { topic, reliable: true });
      } catch (e) {
        // fallback to string publish
        // eslint-disable-next-line no-console
        console.warn('TextEncoder failed when publishing improv_end, sending string fallback', e);
        room.localParticipant.publishData(payload, { topic, reliable: true });
      }
    } catch (err) {
      // non-fatal
      // eslint-disable-next-line no-console
      console.warn('Failed to publish improv_end', err);
    }

    // set local flag and disconnect the room to ensure a fresh connection next time
    setIsSessionActive(false);
    try {
      if (room.state !== 'disconnected') {
        room.disconnect();
      }
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn('Error disconnecting room after endSession', e);
    }
  }, [room]);

  return { room, isSessionActive, startSession, endSession };
}
