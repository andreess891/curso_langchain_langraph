from pydantic import BaseModel

class Usuario(BaseModel):
    id: int
    nombre: str
    activo: bool = True

data = {
    "id": 1,
    "nombre": "Juan Pérez"
}

usuario = Usuario(**data)

print(usuario)