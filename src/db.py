from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URI = "mongodb://root:example@localhost:27017/cash-db?authSource=admin"

client = AsyncIOMotorClient(MONGO_URI)

database = client["cash-db"]
student_collection = database["student"]



async def get_students_by_group(groupId: str) -> list[dict]:
    print(groupId)
    cursor = student_collection.find({"groupId":groupId})
    students = await cursor.to_list(length=None)
    return students
