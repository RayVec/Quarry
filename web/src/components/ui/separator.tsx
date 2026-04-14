"use client"

import * as React from "react"
import { Separator as SeparatorPrimitive } from "radix-ui"
import styles from "./separator.module.css"

import { cn } from "@/lib/utils"

function Separator({
  className,
  orientation = "horizontal",
  decorative = true,
  ...props
}: React.ComponentProps<typeof SeparatorPrimitive.Root>) {
  return (
    <SeparatorPrimitive.Root
      data-slot="separator"
      decorative={decorative}
      orientation={orientation}
      className={cn(styles.root, className)}
      {...props}
    />
  )
}

export { Separator }
