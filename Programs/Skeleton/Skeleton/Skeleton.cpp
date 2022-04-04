//=============================================================================================
// Gravitáló gumilepedõ
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Juhasz Benedek Laszlo
// Neptun : C8B5CT
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

float makerandom() { return float(rand()) / RAND_MAX; }

vec4 qMultiplication(vec4 const& q1, vec4 const& q2) {
	vec3 d1 = vec3(q1.y, q1.z, q1.w);
	vec3 d2 = vec3(q2.y, q2.z, q2.w);
	vec3 d = q1.x * d2 + q2.x * d1 + cross(d1, d2);
	return vec4(
		q1.x * q2.x - dot(d1, d2),
		d.x,
		d.y,
		d.z
	);
}

vec4 qInversion(vec4 const& q) {
	return vec4(q.x, -q.y, -q.z, -q.w) / (dot(q, q) + 0.001f);
}

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec3 startPos;
	Light* other;
	vec4 wLightPos;

	void orbit(float t) {
		vec4 q = vec4(
			cosf(t / 4.0f),
			sinf(t / 4.0f) * cosf(t) / 2.0f,
			sinf(t / 4.0f) * sinf(t) / 2.0f,
			sinf(t / 4.0f) * sqrtf(3.0f / 4.0f)
		);
		vec3 translatedPos = startPos - other->startPos;
		vec4 halfresult = qMultiplication(qMultiplication(q, vec4(0, startPos.x, startPos.y, startPos.z)), qInversion(q));
		wLightPos = vec4(halfresult.y, halfresult.z, halfresult.w, 0) + vec4(other->startPos.x, other->startPos.y, other->startPos.z);
	}
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	vec3 wEye;
};

struct Shader : public GPUProgram {
	virtual void Bind(RenderState const& state) = 0;

	void setUniformMaterial(const Material* material, const std::string& name) {
		if (material != nullptr) {
			setUniform(material->kd, name + ".kd");
			setUniform(material->ks, name + ".ks");
			setUniform(material->ka, name + ".ka");
			setUniform(material->shininess, name + ".shininess");
		}
		else {
			setUniform(vec3(0, 0, 0), name + ".kd");
			setUniform(vec3(0, 0, 0), name + ".ks");
			setUniform(vec3(0, 0, 0), name + ".ka");
			setUniform(0, name + ".shininess");
		}
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class SphereShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++)
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = normalize((vec4(vtxNorm, 0) * Minv).xyz);
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];

        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N,V) < 0)
				N *= -1;
			vec3 radiance = vec3(0,0,0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	SphereShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void Bind(RenderState const& state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(state.material, "material");
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++)
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
	}

};

class SheetShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out float z;
		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++)
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = normalize((vec4(vtxNorm, 0) * Minv).xyz);
			z = wPos.z;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;

		in  float z;
		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];

        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N,V) < 0)
				N *= -1;
			vec3 radiance = vec3(0,0,0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
				vec3 kd = material.kd;
				vec3 ka = material.ka;
				if (z < -0.5){
					kd /= 2;
					ka /= 2;
				}
				if (z < -1){
					kd /= 2;
					ka /= 2;
				}
				if (z < -1.5){
					kd /= 2;
					ka /= 2;
				}
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	SheetShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void Bind(RenderState const& state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(state.material, "material");
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++)
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
	}

};

struct VertexData {
	vec3 position, normal;
};

class Geometry {
public:
	unsigned vao, vbo;
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;

public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void create(int N = 40, int M = 40) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

struct Sphere : public ParamSurface {
	Sphere() {
		create();
	}

	virtual VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vd.normal = vec3(
			cosf(u * 2.0f * M_PI) * sinf(v * M_PI),
			sinf(u * 2.0f * M_PI) * sinf(v * M_PI),
			cosf(v * M_PI));
		return vd;
	}
};

struct Body {
	vec2 position;
	float mass;
};

struct SheetGeometry : public ParamSurface {
	std::vector<Body> bodies;

	const float size;
	const float r0;

	SheetGeometry() : bodies(), size(0.1), r0(size * 0.005) {
		create(80, 80);
	}

	SheetGeometry(std::vector<Body> const& bodies) : bodies(bodies), size(0.1), r0(size * 0.005) {
		create(80, 80);
	}

	float getHeight(float u, float v) {
		float h = 0;
		for (Body b : bodies) {
			float root = length(vec2(u, v) - b.position);
			h -= b.mass / (root + r0);
		}
		return h;
	}

	virtual VertexData GenVertexData(float u, float v) {
		return VertexData{
			vec3(u, v, getHeight(u, v)) ,
			getNormal(u,v)
		};
	}

	vec3 getNormal(float u, float v) {
		float dhdx = 0;
		float dhdy = 0;
		for (Body b : bodies) {
			float root = length(vec2(u, v) - b.position);
			dhdx -= b.mass / (pow(root + r0, 2) * (root + r0)) * (u - b.position.x);
			dhdy -= b.mass / (pow(root + r0, 2) * (root + r0)) * (v - b.position.y);
		}
		return normalize(vec3(
			dhdx,
			dhdy,
			1
		));
	}

	bool hasBody(vec2 const& position) {
		for (Body b : bodies)
			if (length(b.position - position) < 0.05f)
				return true;
		return false;
	}

};

struct Sheet {
	static Shader* shader;
	Material material;
	SheetGeometry* geometry;

	Sheet() {
		material.ks = vec3(0, 0, 0);
		material.ka = vec3(0.2, 0.2, 0.2);
		material.kd = vec3(0.1, 0.2, 0.5);
		material.shininess = 0;
		geometry = new SheetGeometry();
	}

	void addBody(vec2 const& position) {
		static float mass = 0.0f;
		mass += 0.01f;
		std::vector<Body> bodies = geometry->bodies;
		delete geometry;
		bodies.push_back(
			{
				position,
				mass
			}
		);
		geometry = new SheetGeometry(bodies);
	}

	void SetModelingTransformation(RenderState& state) {
		state.M = ScaleMatrix(vec3(1, 1, 1));
		state.Minv = ScaleMatrix(vec3(1, 1, 1));
	}

	void Draw(RenderState state) {
		SetModelingTransformation(state);
		state.MVP = state.M * state.V * state.P;
		state.material = &material;
		shader->Bind(state);
		geometry->Draw();
	}


};

Sheet* sheet;

struct Ball {
	static Geometry* geometry;
	static Shader* shader;
	static const int mass = 1;
	static const int g = 10;

	vec3   position, velocity, acceleration;
	float energy;
	bool   alive;
	bool started;
	Material* material;
	float radius;

	Ball() : position(0.1f, 0.1f, 0), velocity(0, 0, 0), energy(0), acceleration(0, 0, 0), alive(true), started(false), radius(0.03f) {
		material = new Material();
		material->ks = vec3(makerandom(), makerandom(), makerandom());
		material->kd = vec3(makerandom(), makerandom(), makerandom());
		material->shininess = makerandom();
	}
	void Control(float dt) {
		vec3 normal = sheet->geometry->getNormal(position.x, position.y);
		acceleration = vec3(0, 0, -g) - dot(vec3(0, 0, -g), normal) * normal;
	}

	void Animate(float dt) {
		position = position + velocity * dt;
		if (position.x > 1)
			position.x -= 1;
		if (position.y > 1)
			position.y -= 1;
		if (position.x < 0)
			position.x += 1;
		if (position.y < 0)
			position.y += 1;
		velocity = velocity + acceleration * dt;
		if (sheet->geometry->hasBody(vec2(position.x, position.y)))
			alive = false;
		else {
			position.z = sheet->geometry->getHeight(position.x, position.y);
			float requiredVelocity = sqrtf(2.0 * (float(energy) / mass - g * center().z));
			velocity = normalize(velocity) * requiredVelocity;
		}

	}
	void SetModelingTransformation(RenderState & state) {
		state.M = ScaleMatrix(vec3(radius, radius, radius)) * TranslateMatrix(center());
		state.Minv = TranslateMatrix(-center()) * ScaleMatrix(vec3(1 / radius, 1 / radius, 1 / radius));
	}
	void Draw(RenderState state) {
		SetModelingTransformation(state);
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	vec3 center() const {
		return position + sheet->geometry->getNormal(position.x, position.y) * radius;
	}

	void start(vec3 const& velocity) {
		position.z = sheet->geometry->getHeight(position.x, position.y);
		this->velocity = velocity - position;
		energy = mass * (dot(velocity, velocity) / 2.0f + g * center().z);
		started = true;
	}

};

struct Camera {
	vec3 up;
	float fov, asp, fp, bp;

	Camera(vec3 up, float fov, float fp, float bp) :
		up(up),
		fov(fov),
		asp((float)windowWidth / windowHeight),
		fp(fp),
		bp(bp) {}

	virtual mat4 V() = 0;
	virtual vec3 position() = 0;
	virtual mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	virtual Ball const* reference() = 0;
};

struct BallCamera : public Camera {
	Ball const * ball;

	BallCamera(Ball const* ball) :
		Camera(vec3(0, 0, 1), M_PI / 3.0, 0.01f, 3.0f),
		ball(ball) {}

	virtual mat4 V() {
		vec3 w = normalize(-ball->velocity);
		vec3 u = normalize(cross(sheet->geometry->getNormal(ball->position.x, ball->position.y), w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-position()) * mat4(
			u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	virtual vec3 position() {
		return ball->center();
	}

	virtual Ball const* reference() { return ball; }
};

struct StateCamera : public Camera {
	vec3 pos, right, lookat;

	StateCamera() :
		Camera(vec3(0, 1, 0), atanf(0.5 / 100) * 2, 90.0f, 110.0f),
		pos(0.5, 0.5, 100),
		right(1, 0, 0),
		lookat(0.5, 0.5, 0) {}

	virtual mat4 V() {
		vec3 w = normalize(pos - lookat);
		vec3 u = normalize(cross(up, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-pos) * mat4(
			u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	virtual vec3 position() { return pos; }
	virtual Ball const* reference() { return nullptr; }
};

struct Scene {
	std::vector<Ball*> balls;
	std::vector<Light> lights;
	Camera* camera;

	virtual void Build() {
		balls.push_back(new Ball());
		camera = new StateCamera();
		lights.push_back({
			vec3(0.6, 0.6, 0.6),
			vec3(makerandom(), makerandom(), makerandom()),
			vec3(0.5,0.5,0.5)
			});
		lights.push_back({
					vec3(0, 0, 0),
					vec3(makerandom(), makerandom(), makerandom()),
					vec3(-0.3,0.3,0.5)
			});
		lights[0].other = &lights[1];
		lights[1].other = &lights[0];
	}

	void Render() {
		RenderState state;
		state.wEye = camera->position();
		state.V = camera->V();
		state.P = camera->P();
		state.lights = lights;
		Ball const* skipped = camera->reference();
		if (skipped != nullptr && !skipped->alive)
			return;
		sheet->Draw(state);
		for (Ball* ball : balls)
			if (ball != skipped && ball->alive)
				ball->Draw(state);
	}

	void Simulate(float tstart, float tend) {
		const float dt = 0.001f;
		for (Light & l : lights)
			l.orbit(tend);
		for (float t = tstart; t < tend; t += dt) {
			float Dt = fmin(dt, tend - t);
			for (Ball* ball : balls)
				if (ball->alive && ball->started)
					ball->Control(Dt);
			for (Ball* ball : balls)
				if (ball->alive && ball->started)
					ball->Animate(Dt);
		}
	}

	void startBall(vec3 const& velocity) {
		balls.back()->start(velocity);
		balls.push_back(new Ball());
	}

	void addBody(vec2 const& position) {
		sheet->addBody(position);
	}

	void changeCamera() {
		delete camera;
		for (Ball* ball : balls)
			if (ball->alive && ball->started) {
				camera = new BallCamera(ball);
				return;
			}
		camera = new StateCamera();
	}
};

Scene* scene;
Geometry* Ball::geometry;
Shader* Ball::shader;
Shader* Sheet::shader;

void onInitialization() {
	sheet = new Sheet();
	Ball::geometry = new Sphere();
	Ball::shader = new SphereShader();
	Sheet::shader = new SheetShader();

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene = new Scene();
	scene->Build();
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene->Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		scene->changeCamera();
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = float(pX) / windowWidth;
	float cY = 1 - float(pY) / windowHeight;

	if (state == GLUT_UP) {
		switch (button) {
		case GLUT_LEFT_BUTTON:
			scene->startBall(vec3(cX, cY, 0));
			break;
		case GLUT_RIGHT_BUTTON:
			scene->addBody(vec2(cX, cY));
			break;
		}
	}

	glutPostRedisplay();

}

void onIdle() {
	static float tend = 0;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 4000.0f;
	scene->Simulate(tstart, tend);
	glutPostRedisplay();

}
