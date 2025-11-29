import { useMemo } from 'react';
import { Room } from 'livekit-client';
import {
  type ReceivedChatMessage,
  type TextStreamData,
  useChat,
  useRoomContext,
  useTranscriptions,
} from '@livekit/components-react';
import { useGameplayLog } from '@/hooks/useGameplayLog';

function transcriptionToChatMessage(textStream: TextStreamData, room: Room): ReceivedChatMessage {
  return {
    id: textStream.streamInfo.id,
    timestamp: textStream.streamInfo.timestamp,
    message: textStream.text,
    from:
      textStream.participantInfo.identity === room.localParticipant.identity
        ? room.localParticipant
        : Array.from(room.remoteParticipants.values()).find(
            (p) => p.identity === textStream.participantInfo.identity
          ),
  };
}

export function useChatMessages() {
  const chat = useChat();
  const room = useRoomContext();
  const transcriptions: TextStreamData[] = useTranscriptions();
  const gameplayLog = useGameplayLog();

  const mergedTranscriptions = useMemo(() => {
    // Include gameplay log entries (GM / player) so the chat transcript shows backend-written messages
    // without requiring a manual refresh. We map each log entry to a ReceivedChatMessage-like shape.
    const gameplayMessages: Array<ReceivedChatMessage> = (gameplayLog || []).map((e: any, idx: number) => ({
      id: `${e.timestamp}-${e.role}-${idx}`,
      timestamp: new Date(e.timestamp).getTime(),
      message: e.text,
      from: { isLocal: false } as any,
    } as ReceivedChatMessage));

    const merged: Array<ReceivedChatMessage> = [
      ...transcriptions.map((transcription) => transcriptionToChatMessage(transcription, room)),
      ...chat.chatMessages,
      ...gameplayMessages,
    ];
    return merged.sort((a, b) => a.timestamp - b.timestamp);
  }, [transcriptions, chat.chatMessages, room, gameplayLog]);

  return mergedTranscriptions;
}
