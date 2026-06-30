import { StartVideoPipeline } from '../wailsjs/go/main/App';

const canvas = document.getElementById('viewport') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const counterEl = document.getElementById('counter') as HTMLDivElement;

interface TrackedObject {
    id: number;
    box: number[];
    label: string;
}

interface MetaPayload {
    objects?: TrackedObject[];
    totalCount?: number;
}

const COLOR_BY_LABEL: Record<string, string> = {
    person: '#22c55e',
    bicycle: '#eab308',
    car: '#3b82f6',
    motorcycle: '#f97316',
    bus: '#a855f7',
    truck: '#ec4899',
    dog: '#06b6d4',
    cat: '#84cc16',
};
const DEFAULT_COLOR = '#94a3b8';

let latestMeta: MetaPayload = { objects: [], totalCount: 0 };

document.getElementById('connectBtn')!.addEventListener('click', async () => {
    const streamUrl = (document.getElementById('hlsInput') as HTMLInputElement).value;

    try {
        await StartVideoPipeline(streamUrl);
    } catch (err) {
        console.error('StartVideoPipeline failed:', err);
        return;
    }

    const ws = new WebSocket('ws://localhost:8083/stream');
    ws.binaryType = 'arraybuffer';

    ws.onerror = (e) => console.error('WebSocket error:', e);
    ws.onclose = (e) => console.warn('WebSocket closed:', e.code, e.reason);

    ws.onmessage = (msg: MessageEvent) => {
        if (typeof msg.data === 'string') {
            try {
                latestMeta = JSON.parse(msg.data);
            } catch (e) {
                console.error('Failed to parse meta JSON:', e);
            }
            counterEl.textContent = `Objects detected: ${latestMeta.totalCount ?? 0}`;
            return;
        }

        const frameBytes = new Uint8Array(msg.data);
        const imgData = ctx.createImageData(640, 480);
        let dst = 0;
        for (let src = 0; src < frameBytes.length; src += 3) {
            imgData.data[dst]     = frameBytes[src];
            imgData.data[dst + 1] = frameBytes[src + 1];
            imgData.data[dst + 2] = frameBytes[src + 2];
            imgData.data[dst + 3] = 255;
            dst += 4;
        }
        ctx.putImageData(imgData, 0, 0);

        const objects = latestMeta.objects || [];
        objects.forEach(obj => {
            const [x, y, w, h] = obj.box;
            const color = COLOR_BY_LABEL[obj.label] || DEFAULT_COLOR;
            const label = `${obj.label} #${obj.id}`;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = color;
            ctx.fillRect(x, y - 18, Math.max(w, 70), 18);
            ctx.fillStyle = '#0f172a';
            ctx.font = '12px sans-serif';
            ctx.fillText(label, x + 4, y - 5);
        });
    };
});
