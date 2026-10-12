import { readFile } from 'node:fs/promises'
import { log } from '../src/utils/log.js'

log(String(await readFile('package.json', 'utf8')).length)
