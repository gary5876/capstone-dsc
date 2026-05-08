"""webplatform 통합 가이드용 데이터 흐름 다이어그램 PNG 생성.

webplatform v3.2 → v5 통합 후 데이터 흐름. 박스 5개 + 화살표 + 너 작업 위치 ★.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'documents' / 'reports' / 'charts' / 'v5_integration_flow.png'
os.makedirs(OUT.parent, exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')


# ===== 박스 정의 =====
def draw_box(ax, x, y, w, h, label, sublabel='', color='#E8F4F8', edge='#4A90A4',
             star=False, text_size=11):
    """라운드 박스 + 라벨."""
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.05,rounding_size=0.15',
        linewidth=2.5 if star else 1.5,
        edgecolor='#D4632A' if star else edge,
        facecolor='#FFF4E6' if star else color,
    )
    ax.add_patch(box)
    if star:
        ax.text(x - w/2 + 0.1, y + h/2 - 0.15, '★',
                ha='left', va='top', fontsize=14)
    ax.text(x, y + 0.1, label, ha='center', va='center',
            fontsize=text_size, fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.3, sublabel, ha='center', va='center',
                fontsize=text_size - 2, color='#666', style='italic')


# ===== 화살표 정의 =====
def draw_arrow(ax, x1, y1, x2, y2, label='', label_pos=0.5,
               offset=(0, 0), curve=0.0, color='#333', label_color='#222'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f'arc3,rad={curve}',
        arrowstyle='->,head_width=0.35,head_length=0.5',
        linewidth=1.8, color=color,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = x1 + (x2 - x1) * label_pos + offset[0], y1 + (y2 - y1) * label_pos + offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCC', linewidth=0.5),
                color=label_color)


# ===== 다이어그램 그리기 =====

# 제목
ax.text(7, 8.5, 'aidq-platform v5 통합 — 데이터 흐름',
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(7, 8.05, '★ 표시 = 웹 백엔드 측 작업 영역',
        ha='center', va='center', fontsize=10, color='#D4632A')

# 박스 5개 (지그재그 레이아웃)
draw_box(ax, 1.8, 6.5, 2.6, 1.0, 'React 브라우저', '사용자 UI', star=True)
draw_box(ax, 6.0, 6.5, 2.6, 1.0, 'Spring Boot API', '업로드·인증·MQ 발행')
draw_box(ax, 11.0, 6.5, 3.4, 1.0, 'Python Worker', 'dsc_framework 사용', star=True)
draw_box(ax, 11.0, 3.0, 3.4, 1.0, 'Result Listener', 'Spring Boot', star=True)
draw_box(ax, 6.0, 3.0, 2.6, 1.0, 'MySQL + S3', 'resultDetail JSON 저장')

# 화살표 + 라벨
# ① React → API
draw_arrow(ax, 3.1, 6.5, 4.7, 6.5,
           label='① CSV 업로드', label_pos=0.5, offset=(0, 0.35))

# ② API → Worker (RabbitMQ)
draw_arrow(ax, 7.3, 6.5, 9.3, 6.5,
           label='② diagnosis.queue', label_pos=0.5, offset=(0, 0.35))

# ③ Worker → Listener (결과 큐, task 포함 ★)
draw_arrow(ax, 11.0, 6.0, 11.0, 3.5,
           label='③ result.queue\n(task 필드 포함 ★)', label_pos=0.5,
           offset=(1.3, 0), label_color='#D4632A')

# ④ Listener → DB
draw_arrow(ax, 9.3, 3.0, 7.3, 3.0,
           label='④ DB 저장', label_pos=0.5, offset=(0, 0.35))

# ⑤ DB → React (조회)
draw_arrow(ax, 5.5, 3.5, 1.8, 6.0,
           label='⑤ 결과 조회\n+ task 배지', label_pos=0.65, offset=(-0.3, 0.0),
           curve=-0.25, label_color='#D4632A')


# ===== Worker 안 zoom-in (오른쪽 박스 안에 무엇이 일어나는지) =====
zoom_x, zoom_y, zoom_w, zoom_h = 11.0, 0.6, 5.5, 1.4
zoom = patches.FancyBboxPatch(
    (zoom_x - zoom_w/2, zoom_y - zoom_h/2), zoom_w, zoom_h,
    boxstyle='round,pad=0.05,rounding_size=0.1',
    linewidth=1.5, edgecolor='#D4632A', facecolor='#FFF8F0',
    linestyle='--',
)
ax.add_patch(zoom)
ax.text(zoom_x - zoom_w/2 + 0.15, zoom_y + zoom_h/2 - 0.2, 'Worker 내부 변경 ★',
        ha='left', va='top', fontsize=10, fontweight='bold', color='#D4632A')
ax.text(zoom_x - zoom_w/2 + 0.3, zoom_y + 0.15,
        '• auto_detect_columns(df) → 4-tuple (target, num, cat, task)',
        ha='left', va='center', fontsize=9)
ax.text(zoom_x - zoom_w/2 + 0.3, zoom_y - 0.2,
        '• compute_dsc(df, ..., task=task)',
        ha='left', va='center', fontsize=9)
ax.text(zoom_x - zoom_w/2 + 0.3, zoom_y - 0.55,
        "• result_detail에 'task' 필드 포함",
        ha='left', va='center', fontsize=9)

# Worker → zoom 점선 연결
ax.plot([11.0, 11.0], [2.5, 1.4], linestyle=':', color='#D4632A', linewidth=1)


# ===== React 안 zoom-in (왼쪽 박스 아래) =====
left_x, left_y, left_w, left_h = 2.5, 0.9, 4.5, 1.0
left_zoom = patches.FancyBboxPatch(
    (left_x - left_w/2, left_y - left_h/2), left_w, left_h,
    boxstyle='round,pad=0.05,rounding_size=0.1',
    linewidth=1.5, edgecolor='#D4632A', facecolor='#FFF8F0',
    linestyle='--',
)
ax.add_patch(left_zoom)
ax.text(left_x - left_w/2 + 0.15, left_y + left_h/2 - 0.2, 'React 변경 ★',
        ha='left', va='top', fontsize=10, fontweight='bold', color='#D4632A')
ax.text(left_x - left_w/2 + 0.3, left_y - 0.05,
        '• task 배지 표시 (분류 / 회귀)',
        ha='left', va='center', fontsize=9)
ax.text(left_x - left_w/2 + 0.3, left_y - 0.35,
        '• Slider 9개 (task별 키 셋 분기)',
        ha='left', va='center', fontsize=9)

# React → left_zoom 점선
ax.plot([1.8, 2.5], [6.0, 1.4], linestyle=':', color='#D4632A', linewidth=1)


# 범례
legend_y = 7.5
ax.text(0.5, legend_y, '범례:', fontsize=9, fontweight='bold')
# 일반 박스 샘플
sample1 = patches.FancyBboxPatch((1.3, legend_y - 0.15), 0.4, 0.3,
                                  boxstyle='round,pad=0.02,rounding_size=0.05',
                                  linewidth=1.2, edgecolor='#4A90A4',
                                  facecolor='#E8F4F8')
ax.add_patch(sample1)
ax.text(1.85, legend_y, '변경 없음', fontsize=8, va='center')
# ★ 박스 샘플
sample2 = patches.FancyBboxPatch((3.5, legend_y - 0.15), 0.4, 0.3,
                                  boxstyle='round,pad=0.02,rounding_size=0.05',
                                  linewidth=2, edgecolor='#D4632A',
                                  facecolor='#FFF4E6')
ax.add_patch(sample2)
ax.text(4.05, legend_y, '★ 너 작업 영역', fontsize=8, va='center')


plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

print(f'생성: {OUT}')
print(f'크기: {os.path.getsize(OUT) / 1024:.1f} KB')
