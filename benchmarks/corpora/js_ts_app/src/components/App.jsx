import Button from './Button'

const lazy = () => import('./Lazy')

export const App = () => <Button label={String(lazy)} />
