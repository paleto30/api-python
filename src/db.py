from motor.motor_asyncio import AsyncIOMotorClient




MONGO_URI = "mongodb://root:example@localhost:27017/cash-db?authSource=admin"


client = AsyncIOMotorClient(MONGO_URI)


database = client["cash-db"]
user_collection = database["user"]