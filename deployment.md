# This is back end for g-nayan RUNNING IN FAST API 
docker-compose up --build
# To build image in a best way and quick way we can use COMPOSE_BAKE=true 
COMPOSE_BAKE=true docker-compose up --build 

# To connect mysql image
docker exec -it mysql-container mysql -u root -p
with password 