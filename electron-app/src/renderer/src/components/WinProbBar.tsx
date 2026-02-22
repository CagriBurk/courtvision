interface WinProbBarProps {
  probHome: number
  homeColor: string
  awayColor: string
}

export default function WinProbBar({ probHome, homeColor, awayColor }: WinProbBarProps) {
  return (
    <div className="h-2 rounded-full overflow-hidden flex gap-px">
      <div
        className="h-full rounded-l-full transition-all duration-500"
        style={{ width: `${probHome * 100}%`, backgroundColor: homeColor }}
      />
      <div
        className="h-full rounded-r-full flex-1 transition-all duration-500"
        style={{ backgroundColor: awayColor }}
      />
    </div>
  )
}
