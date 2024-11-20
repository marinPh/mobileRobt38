from tdmclient import ClientAsync, aw


class Thymio:
    async def __init__(self):
        self.client = ClientAsync()
        self.node = await self.client.wait_for_node()
        await self.node.lock()
        

    async def lock_node(self):
        await self.node.lock()

    def wait_for_variables(self, variables):
        aw(self.node.wait_for_variables(variables))

    async def sleep(self, duration):
        await self.client.sleep(duration)
        
    async def set_var(self, var, value):
        await self.node.set_var(var, value)
        
    async def getProxH(self):
        self.wait_for_variables(["prox.horizontal"])
        aw(self.client.sleep(0.1))
        return self.node.v.prox.horizontal
    
    async def getWheelR(self):
        self.wait_for_variables(["motor.right.speed"])
        aw(self.client.sleep(0.1))
        return self.node.v.motor.right.speed
    
    async def getWheelL(self):
        self.wait_for_variables(["motor.left.speed"])
        aw(self.client.sleep(0.1))
        return self.node.v.motor.left.speed