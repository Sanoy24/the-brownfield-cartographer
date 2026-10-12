export interface Config {
  port: number
}

export const readConfig = (): Config => ({ port: Number(process.env.PORT ?? 3000) })
