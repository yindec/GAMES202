#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

//遮挡物平均深度的计算
float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  poissonDiskSamples(uv);
  float totalDepth=0.0;
  int blockCount =0;

  for(int i=0;i<NUM_SAMPLES; i ++){
    vec2 simpleUV = uv +poissonDisk[i]/float(2048) *50.0;
    float shadowMapDepth = unpack(vec4(texture2D(uShadowMap,simpleUV).rgb,1.0));
    if(zReceiver > (shadowMapDepth+0.003)){
      totalDepth += shadowMapDepth;
      blockCount +=1;
    }
  }


  //没有遮挡
  if(blockCount ==0){
    return -1.0;
  }

  //完全遮挡
  if(blockCount==NUM_SAMPLES){
    return 2.0;
  }

	return totalDepth/float( blockCount );
}

float PCF(sampler2D shadowMap, vec4 coords) {
  float currentDepth = coords.z;
  float visibility = 0.0;
  poissonDiskSamples(coords.xy);
  // shadow map 的大小， 越大滤波的范围越小
  float textureSize = 2048.0;
  // 滤波的步长
  float filterStride = 5.0;
  // 滤波窗口的范围
  float filterRange = filterStride / textureSize;

  for(int i = 0; i < NUM_SAMPLES; i++){
    vec4 closestDepthVec = texture2D(shadowMap, poissonDisk[i]* filterRange + coords.xy);
    float closestDepth = unpack(closestDepthVec);
    if(currentDepth < closestDepth + 0.01){
      visibility += 1.0 / float(NUM_SAMPLES);
    }
  }
  return visibility;
}

//利用相似三角形计算半影直径并传递给 PCF 函数以调整其滤波核大小
float PCSS(sampler2D shadowMap, vec4 coords){

  // STEP 1: avgblocker depth 平均遮挡深度
   float zBlocker=findBlocker(shadowMap,coords.xy,coords.z);
  if(zBlocker<EPS){//没有被遮挡
    return 1.0;
  }

  if(zBlocker >1.0+EPS){
    return 0.0;
  }

  // STEP 2: penumbra size 确定半影的大小
  float penumbraScale =(coords.z -zBlocker) /zBlocker;


  // STEP 3: filtering  过滤
  return PCF_Internal(shadowMap,coords.xy,coords.z,penumbraScale*10.0);

}


float Bias(){   //会导致漏光
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.01);
  return bias;
}

float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){

  // Edit Start

  float lightDepth = unpack(texture2D(shadowMap, shadowCoord.xy));
  float shadingDepth = shadowCoord.z;
  float bias = Bias();
  return lightDepth + EPS <= shadingDepth - bias ? 0.2 : 0.9;
  
  // Edit End
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  // Edit Start
  
  //vPositionFromLight为光源空间下投影的裁剪坐标，除以w结果为NDC坐标
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  //把[-1,1]的NDC坐标转换为[0,1]的坐标
  shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;

  float visibility;
  

  // Edit End

  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}