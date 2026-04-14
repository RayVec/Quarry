import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"
import styles from "./badge.module.css"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  styles.root,
  {
    variants: {
      variant: {
        default: styles.variantDefault,
        secondary: styles.variantSecondary,
        destructive: styles.variantDestructive,
        outline: styles.variantOutline,
        ghost: styles.variantGhost,
        link: styles.variantLink,
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
