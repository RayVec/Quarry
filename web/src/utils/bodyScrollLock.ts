let lockCount = 0;
let previousOverflow = "";
let previousPaddingRight = "";

function scrollbarWidthPx(): number {
  return Math.max(0, window.innerWidth - document.documentElement.clientWidth);
}

export function lockBodyScroll(): void {
  if (lockCount === 0) {
    previousOverflow = document.body.style.overflow;
    previousPaddingRight = document.body.style.paddingRight;
    const gap = scrollbarWidthPx();
    document.body.style.overflow = "hidden";
    if (gap > 0) {
      document.body.style.paddingRight = `${gap}px`;
    }
  }
  lockCount += 1;
}

export function unlockBodyScroll(): void {
  lockCount = Math.max(0, lockCount - 1);
  if (lockCount === 0) {
    document.body.style.overflow = previousOverflow;
    document.body.style.paddingRight = previousPaddingRight;
  }
}
