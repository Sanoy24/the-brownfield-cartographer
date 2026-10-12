import express from 'express'
import { readConfig } from './config.js'
import { log } from '@/utils/log'

export function createServer() {
  const config = readConfig()
  log(`listening on ${config.port}`)
  return express()
}

export class Server {
  start() {
    return createServer()
  }

  stop() {
    log('stopped')
  }
}
