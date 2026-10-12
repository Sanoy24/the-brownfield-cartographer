import { log } from '../utils'

export default function Button({ label }: { label: string }) {
  return <button onClick={() => log(label)}>{label}</button>
}
