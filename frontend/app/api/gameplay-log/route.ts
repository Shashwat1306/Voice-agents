import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import { join } from 'path';

export async function GET() {
  try {
    const root = process.cwd();
    const filePath = join(root, 'backend', 'shared-data', 'gameplay_log.json');
    const raw = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(raw || '[]');
    return NextResponse.json({ ok: true, data });
  } catch (err) {
    return NextResponse.json({ ok: true, data: [] });
  }
}
