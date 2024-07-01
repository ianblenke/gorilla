up: build
	docker compose up --force-recreate -d
	docker compose logs -f

down:
	docker compose down

build:
	cog build -t gorilla
	jq . .cog/openapi_schema.json > .cog/openapi_schema.json.new
	mv -f .cog/openapi_schema.json.new .cog/openapi_schema.json
