const { createServer } = require('../src/index')
const plugin = process.env.BUILD_PLUGIN
if (plugin) require(plugin)
module.exports = createServer
