#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_shuffledPos;
glm::vec3 *dev_shuffledVel;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

float avgTimeTaken = 0.0f;
float countTimeTaken = 0;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2*std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
	
	cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");
	
	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");
	
	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

	dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

	cudaMalloc((void**)&dev_shuffledPos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_shuffledVel, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  
	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	// Rule 2: boids try to stay a distance d away from each other
	// Rule 3: boids try to match the speed of surrounding boids
	
	// Rule 1
	glm::vec3 finalDir(0.0f, 0.0f, 0.0f);
	int numOfNeighbours = 0;

	for (int i = 0; i < N; i++) {
		if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule1Distance) {
			finalDir += pos[i];
			numOfNeighbours += 1;
		}
	}

	if (numOfNeighbours != 0) {
		finalDir /= numOfNeighbours;
		finalDir = (finalDir - pos[iSelf]) * rule1Scale;
	}

	// Rule 2
	glm::vec3 finalDir2(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < N; i++) {
		if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule2Distance) {
			finalDir2 -= (pos[i] - pos[iSelf]);
		}
	}

	finalDir2 = finalDir2 * rule2Scale;

	glm::vec3 finalVel(0.0f, 0.0f, 0.0f); 
	numOfNeighbours = 0;
	for (int i = 0; i < N; i++) {
		if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule3Distance) {
			finalVel += vel[i];
			numOfNeighbours += 1;
		}
	}

	if (numOfNeighbours != 0) {
		finalVel /= numOfNeighbours;
		finalVel = finalVel * rule3Scale;
	}

	finalVel = finalVel + finalDir + finalDir2;

  return finalVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

	int boidId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (boidId >= N) return;

	glm::vec3 finalVel = computeVelocityChange(N, boidId, pos, vel1);
	vel2[boidId] = glm::clamp(vel1[boidId] + finalVel, -maxSpeed, maxSpeed);
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
  // TODO-2.1
  //-Label each boid with the index of its grid cell.
  //-Set up a parallel array of integer indices as pointers to the actual
  // boid data in pos and vel1/vel2

	int boidId = threadIdx.x + (blockIdx.x * blockDim.x);

	if (boidId >= N) return;

	glm::vec3 grid3d = glm::floor((pos[boidId] - gridMin) * inverseCellWidth);

	int gridId = gridIndex3Dto1D((int) grid3d.x, (int) grid3d.y, (int) grid3d.z, gridResolution);

	indices[boidId] = boidId;
	gridIndices[boidId] = gridId;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
	int *gridCellStartIndices, int *gridCellEndIndices) {
	// TODO-2.1
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	//float x = p[index].x, y = p[index].y, z = p[index].z;

	if(index != 0 && particleGridIndices[index] != particleGridIndices[index-1]) {
		gridCellEndIndices[particleGridIndices[index-1]] = index-1;
		gridCellStartIndices[particleGridIndices[index]] = index;
	}
	else if (index == 0) {
		gridCellStartIndices[particleGridIndices[0]] = 0;
	}
	else if (index == N - 1) {
		gridCellEndIndices[particleGridIndices[N - 1]] = N - 1;
	}

}


__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (iSelf >= N) return;

	int countN1 = 0, countN3 = 0;

	glm::vec3 finalVel(0.0f, 0.0f, 0.0f);
	glm::vec3 finalDir(0.0f, 0.0f, 0.0f);
	glm::vec3 finalCenter(0.0f, 0.0f, 0.0f);

	float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);

	glm::vec3 grid3d = glm::floor((pos[iSelf] - gridMin) * inverseCellWidth);
	glm::vec3 grid3d_min = glm::floor((pos[iSelf] - gridMin - maxDistance) * inverseCellWidth);
	glm::vec3 grid3d_max = glm::floor((pos[iSelf] - gridMin + maxDistance) * inverseCellWidth);

	int x_g = (int)grid3d.x, y_g = (int)grid3d.y, z_g = (int)grid3d.z;
	int x_g_min = (int)grid3d_min.x, y_g_min = (int)grid3d_min.y, z_g_min = (int)grid3d_min.z;
	int x_g_max = (int)grid3d_max.x, y_g_max = (int)grid3d_max.y, z_g_max = (int)grid3d_max.z;

	x_g_max = imin(x_g_max, gridResolution - 1), y_g_max = imin(y_g_max, gridResolution - 1), z_g_max = imin(z_g_max, gridResolution - 1);
	x_g_min = imax(x_g_min, 0), y_g_min = imax(y_g_min, 0), z_g_min = imax(z_g_min, 0);

	//x_g_max = x_g-1, y_g_max = x_g-1, z_g_max = imin(z_g_max, gridResolution - 1);
	//x_g_min = imax(x_g_min, 0), y_g_min = imax(y_g_min, 0), z_g_min = imax(z_g_min, 0);


	for (int x = x_g_min; x <= x_g_max; x++) {
		for (int y = y_g_min; y <= y_g_max; y++) {
			for (int z = z_g_min; z <= z_g_max; z++) {
				if (x >= 0 && y >= 0 && z >= 0 && x < gridResolution && y < gridResolution && z < gridResolution) {
					int gridId = gridIndex3Dto1D(x, y, z, gridResolution);
					
					int start = gridCellStartIndices[gridId];
					int end = gridCellEndIndices[gridId];
					if (start < 0 || start >= N || end < 0 || end >= N) continue;

					for (int idx = start; idx <= end; idx++) {
						int bid = particleArrayIndices[idx];

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule1Distance) {
							finalCenter += pos[bid];
							countN1 += 1;
						}

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule2Distance) {
							finalDir -= (pos[bid] - pos[iSelf]);
						}

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule3Distance) {
							finalVel += vel1[bid];
							countN3 += 1;
						}
					}
				}
			}
		}
	}

	if (countN1 != 0) {
		finalCenter /= countN1;
		finalCenter = (finalCenter - pos[iSelf]) * rule1Scale;
	}

	finalDir = finalDir * rule2Scale;

	if (countN3 != 0) {
		finalVel /= countN3;
		finalVel = finalVel * rule3Scale;
	}

	glm::vec3 resultVel = finalCenter + finalDir + finalVel;
	vel2[iSelf] = glm::clamp(vel1[iSelf] + resultVel, -maxSpeed, maxSpeed);
}



__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (iSelf >= N) return;

	int countN1 = 0, countN3 = 0;

	glm::vec3 finalVel(0.0f, 0.0f, 0.0f);
	glm::vec3 finalDir(0.0f, 0.0f, 0.0f);
	glm::vec3 finalCenter(0.0f, 0.0f, 0.0f);

	float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);
	
	glm::vec3 grid3d = glm::floor((pos[iSelf] - gridMin) * inverseCellWidth); // cell coordinates of the boid
	glm::vec3 grid3d_min = glm::floor((pos[iSelf] - gridMin - maxDistance) * inverseCellWidth);
	glm::vec3 grid3d_max = glm::floor((pos[iSelf] - gridMin + maxDistance) * inverseCellWidth);

	int x_g = (int)grid3d.x, y_g = (int)grid3d.y, z_g = (int)grid3d.z;
	int x_g_min = (int)grid3d_min.x, y_g_min = (int)grid3d_min.y, z_g_min = (int)grid3d_min.z;
	int x_g_max = (int)grid3d_max.x, y_g_max = (int)grid3d_max.y, z_g_max = (int)grid3d_max.z;

	x_g_max = imin(x_g_max, gridResolution - 1), y_g_max = imin(y_g_max, gridResolution - 1), z_g_max = imin(z_g_max, gridResolution - 1);
	x_g_min = imax(x_g_min, 0), y_g_min = imax(y_g_min, 0), z_g_min = imax(z_g_min, 0);

	for (int x = x_g_min; x <= x_g_max; x++) {
		for (int y = y_g_min; y <= y_g_max; y++) {
			for (int z = z_g_min; z <= z_g_max; z++) {
				if (x >= 0 && y >= 0 && z >= 0 && x < gridResolution && y < gridResolution && z < gridResolution) {
					int gridId = gridIndex3Dto1D(x, y, z, gridResolution);

					int start = gridCellStartIndices[gridId];
					int end = gridCellEndIndices[gridId];
					if (start < 0 || start >= N || end < 0 || end >= N) continue;

					for (int idx = start; idx <= end; idx++) {
						int bid = idx;

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule1Distance) {
							finalCenter += pos[bid];
							countN1 += 1;
						}

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule2Distance) {
							finalDir -= (pos[bid] - pos[iSelf]);
						}

						if (bid != iSelf && glm::distance(pos[iSelf], pos[bid]) < rule3Distance) {
							finalVel += vel1[bid];
							countN3 += 1;
						}
					}
				}
			}
		}
	}

	if (countN1 != 0) {
		finalCenter /= countN1;
		finalCenter = (finalCenter - pos[iSelf]) * rule1Scale;
	}

	finalDir = finalDir * rule2Scale;

	if (countN3 != 0) {
		finalVel /= countN3;
		finalVel = finalVel * rule3Scale;
	}

	glm::vec3 resultVel = finalCenter + finalDir + finalVel;
	vel2[iSelf] = glm::clamp(vel1[iSelf] + resultVel, -maxSpeed, maxSpeed);
}


__global__ void kernShufflePosAndVel(int N, glm::vec3 *pos, glm::vec3 *vel, glm::vec3 *shuffledPos, 
		glm::vec3 *shuffledVel, int *particleArrayIndices) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	shuffledPos[index] = pos[particleArrayIndices[index]]; // now shuffledPos[index] is the position of boid at particleArrayIndices[i]
	shuffledVel[index] = vel[particleArrayIndices[index]];
}

__global__ void kernUnShuffleVel(int N, glm::vec3 *vel ,glm::vec3 *shuffledVel, int *particleArrayIndices) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) return;

	vel[particleArrayIndices[index]] = shuffledVel[index];
}


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	dim3 numBlocks((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce<<<numBlocks, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);

	checkCUDAErrorWithLine("Velocity update failed");

	kernUpdatePos<<<numBlocks, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

	checkCUDAErrorWithLine("Position update failed");

	dev_vel1 = dev_vel2;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	avgTimeTaken += milliseconds;
	if (countTimeTaken == 100){
		avgTimeTaken /= countTimeTaken;
		printf("Time taken for one simulation step: %.2f\n", avgTimeTaken);
	}
	countTimeTaken++;
}




void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

	dim3 numBlocks((numObjects + blockSize - 1) / blockSize);
	int N = numObjects;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//int threadsPerBlockCP = 2 * blockSize;
	//dim3 numBlocksCellPointers((gridCellCount + threadsPerBlockCP - 1) / threadsPerBlockCP);

	kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	//kernResetIntBuffer<<<numBlocksCellPointers, threadsPerBlockCP>>>(gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("Cell start reset failed");
	kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	//kernResetIntBuffer<<<numBlocksCellPointers, threadsPerBlockCP >>>(gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("Cell end reset failed");

	kernComputeIndices<<<numBlocks, blockSize>>>(N, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("Compute indices failed");

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

	kernIdentifyCellStartEnd<<<numBlocks, blockSize>>>(N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("Cell start/end failed");

	kernUpdateVelNeighborSearchScattered<<<numBlocks, blockSize>>>(N, gridSideCount, gridMinimum, 
		gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices,
		dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("Velocity update failed");

	kernUpdatePos<<<numBlocks, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("Position update failed");

	dev_vel1 = dev_vel2;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	avgTimeTaken += milliseconds;
	if (countTimeTaken == 1000) {
		avgTimeTaken /= countTimeTaken;
		printf("Time taken for one simulation step: %.2f\n", avgTimeTaken);
	}
	countTimeTaken++;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	dim3 numBlocks((numObjects + blockSize - 1) / blockSize);
	int N = numObjects;

	//int threadsPerBlockCP = 2 * blockSize;
	//dim3 numBlocksCellPointers((gridCellCount + threadsPerBlockCP - 1) / threadsPerBlockCP);

	kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("Cell start reset failed");
	kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("Cell end reset failed");

	kernComputeIndices << <numBlocks, blockSize >> > (N, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("Compute indices failed");

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

	kernIdentifyCellStartEnd << <numBlocks, blockSize >> > (N, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("Cell start/end failed");

	// shuffle position and velocity
	kernShufflePosAndVel<< <numBlocks, blockSize >> > (N, dev_pos, dev_vel1, dev_shuffledPos, dev_shuffledVel, dev_particleArrayIndices);

	kernUpdateVelNeighborSearchCoherent<< <numBlocks, blockSize >> > (N, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices,
		dev_gridCellEndIndices, dev_shuffledPos, dev_shuffledVel, dev_vel2);
	checkCUDAErrorWithLine("Velocity update failed");

	// unshuffle velocity and put it back in dev_vel1
	kernUnShuffleVel<< <numBlocks, blockSize>>>(N, dev_vel1, dev_vel2, dev_particleArrayIndices);

	kernUpdatePos << <numBlocks, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("Position update failed");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	avgTimeTaken += milliseconds;
	if (countTimeTaken == 1000) {
		avgTimeTaken /= countTimeTaken;
		printf("Time taken for one simulation step: %.2f\n", avgTimeTaken);
	}
	countTimeTaken += 1;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);

	cudaFree(dev_shuffledPos);
	cudaFree(dev_shuffledVel);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
