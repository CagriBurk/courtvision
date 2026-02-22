import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
} from 'recharts'

interface FormChartProps {
  homeForm: number[]
  awayForm: number[]
  homeColor: string
  awayColor: string
  homeAbbr: string
  awayAbbr: string
}

export default function FormChart({
  homeForm,
  awayForm,
  homeColor,
  awayColor,
  homeAbbr,
  awayAbbr,
}: FormChartProps) {
  const data = homeForm.map((h, i) => ({
    game: i + 1,
    [homeAbbr]: h,
    [awayAbbr]: awayForm[i],
  }))

  return (
    <div className="h-28">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -28 }}>
          <XAxis
            dataKey="game"
            tick={{ fontSize: 9, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[-0.2, 1.2]}
            ticks={[0, 1]}
            tickFormatter={(v) => (v === 1 ? 'G' : 'M')}
            tick={{ fontSize: 9, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine y={0.5} stroke="#334155" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: 8,
              fontSize: 11,
              color: '#e2e8f0',
            }}
            formatter={(v, name) => [(v ?? 0) === 1 ? 'Galibiyet' : 'Mağlubiyet', name as string]}
          />
          <Line
            type="monotone"
            dataKey={homeAbbr}
            stroke={homeColor}
            strokeWidth={2}
            dot={{ r: 3, fill: homeColor, strokeWidth: 0 }}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey={awayAbbr}
            stroke={awayColor}
            strokeWidth={2}
            strokeDasharray="4 2"
            dot={{ r: 3, fill: awayColor, strokeWidth: 0 }}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
