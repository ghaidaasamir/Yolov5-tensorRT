#ifndef __TENSORRT_UTILS__
#define __TENSORRT_UTILS__

#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
#include <sstream> 
#include <NvInfer.h>


#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}




namespace TRT
{




inline void getGPUinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}



// inline void checkCudaErrors(cudaError_t status)                                   
// {                                                                 
//   if (status != 0)                                                
//   {                                                               
//     std::cout << "Cuda failure: " << cudaGetErrorString(status)   
//               << " at line " << __LINE__                          
//               << " in file " << __FILE__                          
//               << " error status: " << status                      
//               << std::endl;                                       
//               abort();                                            
//     }                                                             
// }


class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


inline void printDim(const nvinfer1::Dims & dims)
{
  auto dim_num = dims.nbDims ;
  for (int i = 0 ;i<dim_num ; i++)
  {
    std::cout << dims.d[i] << " ," ;
  }
  std::cout << "  " << std::endl;

}


inline size_t getMemorySize(const nvinfer1::Dims & dims, const int32_t elem_size)
{
  int dim_num = dims.nbDims ;
  int dim = 1 ;
  for (int i = 0 ;i<dim_num ; i++)
  {
    dim = dim * dims.d[i] ;
  }

  size_t size = sizeof(elem_size) * dim ;
  return size; 
}



struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if(obj != nullptr)
        {
            delete obj;
        }
    }
};

template <typename T>
using uniquePtr = std::unique_ptr<T, InferDeleter>;


template <typename T>
uniquePtr<T> makeUnique(T* t)
{
    return uniquePtr<T>{t};
}



inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}


inline int64_t volume(nvinfer1::Dims const& d)
{
    int64_t vol = 1 ;
    for(int i ; i<d.nbDims ; i++ )
    {
        vol *= d.d[i] ;
    }
    // std::cout << "vol = " << vol << std::endl ;
    return abs(vol) ; 
    // return std::accumulate(d.d , d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

inline void getContextInfo(std::shared_ptr<nvinfer1::IExecutionContext> context , int bindingNum )
{
    for (int i = 0; i <bindingNum; i++)
    {
        const char * binding_name = (context->getEngine()).getIOTensorName(i) ;
        auto dims =  context->getTensorShape(binding_name);        
        
        auto dim_num = dims.nbDims ;
        std::stringstream dimsShape;
        dimsShape << "(" ;
        for (int i = 0 ;i<dim_num ; i++)
        {
            dimsShape << dims.d[i] << " ,";
        }
        dimsShape << ")" ;
        std::cout << "binding num " << i << " shape = " << dimsShape.str() << std::endl ; 
    }
}


inline void getEngineInfo(std::shared_ptr<nvinfer1::ICudaEngine> const engine)
{
    for (int i = 0; i < engine->getNbIOTensors() ; i++)
    {
        const char * binding_name = engine->getIOTensorName(i);
        auto dims =  engine->getTensorShape(binding_name);
        auto dim_num = dims.nbDims ;
        std::stringstream dimsShape;
        dimsShape << "(" ;
        for (int i = 0 ;i<dim_num ; i++)
        {
            dimsShape << dims.d[i] << " ,";
        }
        dimsShape << ")" ;

        size_t bindingMemSize = getMemorySize(dims , sizeof(float));
        std::string bindingType ;
        if(engine->getTensorIOMode(binding_name) ==  nvinfer1::TensorIOMode::kINPUT)
        {
            bindingType = "input" ;
        }
        else 
        {
            bindingType = "output" ;
        }
        std::cout << "binding Num " << i  << " Name " << binding_name <<  " is " << bindingType << " Size = " << bindingMemSize << " shape = " << dimsShape.str() << std::endl ;
    } 
}  


/**
 * @brief The GenericBuffer class is a templated class for buffers.
 *        details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
 *        deallocation, querying of buffers on both the device and the host.
 *        It can handle data of arbitrary types because it stores byte buffers.
 *        The template parameters AllocFunc and FreeFunc are used for the
 *        allocation and deallocation of the buffer.
 *        AllocFunc must be a functor that takes in (void** ptr, size_t size)
 *        and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
 *        size is the amount of memory in bytes to allocate.
 *        The boolean indicates whether or not the memory allocation was successful.
 *        FreeFunc must be a functor that takes in (void* ptr) and returns void.
 *        ptr is the allocated buffer address. It must work with nullptr input.
 *
 *  @tparam AllocFunc 
 * @tparam FreeFunc 
 */
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:

    /**
     * @brief Construct a new Generic Buffer object
     * 
     * @param type 
     */
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
        // std::cout << "GenericBuffer Constructor " << std::endl ; 
    }

    /**
     * @briefConstruct a buffer with the specified allocation size in bytes.
     * 
     * @param size 
     * @param type 
     */
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            // std::cout << "throw std::bad_alloc() hereeeeee" << std::endl ; 
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }


    /**
     * @brief  Returns pointer to underlying array.
     * 
     * @return void* 
     */
    void* data()
    {
        return mBuffer;
    }

    /**
     * @brief Returns pointer to underlying array.
     * 
     * @return const void* 
     */
    const void* data() const
    {
        return mBuffer;
    }

    /**
     * @brief Returns the size (in number of elements) of the buffer.
     * 
     * @return size_t 
     */
    size_t size() const
    {
        return mSize;
    }


    /**
     * @brief Returns the size (in bytes) of the buffer.
     * 
     * @return size_t 
     */
    size_t nbBytes() const
    {
        return this->size() * getElementSize(mType);
    }

    /**
     * @brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
     * 
     * @param newSize 
     */
    void resize(size_t newSize)
    {
        std::cout << "************* resize call **************" << std::endl ;
        std::cout << "Size = "  << mSize << std::endl ;
        std::cout << "Capacity = "  << mCapacity << std::endl ;
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if ( !allocFn(&mBuffer, this->nbBytes()) )
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;

        }
        std::cout << "new Size = "  << mSize << std::endl ;
        std::cout << "new Capacity = "  << mCapacity << std::endl ;
        std::cout << "************** resize end *************" << std::endl ;
    }


    /**
     * @brief Overload of resize that accepts Dims
     * 
     * @param dims 
     */
    void resize(const nvinfer1::Dims& dims)
    {
        // std::cout<< "volume = " << volume(dims) << std::endl ; 
        return this->resize(volume(dims));
    }

    ~GenericBuffer()
    {
        // std::cout<< "~GenericBuffer" << std::endl;
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        // std::cout << "***************** DeviceAllocator *****************" << std::endl ; 
        return cudaMalloc(ptr, size) == cudaError::cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        // std::cout << "*************** HostAllocator *****************" << std::endl ; 
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;


/**
 * @brief The ManagedBuffer class groups together a pair of corresponding device and host buffers.
 * 
 */
class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;

    ~ManagedBuffer()
    {
        // std::cout<< "~ManagedBuffer" << std::endl;
    }

};


/**
 * @brief The BufferManager class handles host and device buffer allocation and deallocation.
 *        This RAII class handles host and device buffer allocation and deallocation,
 *        memcpy between host and device buffers to aid with inference,
 *        and debugging dumps to validate inference. The BufferManager class is meant to be
 *        used to simplify buffer management and any interactions between buffers and the engine.
 * 
 */
class BufferManager
{
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    BufferManager()
    {

    }

    // const char* dataTypeToString(nvinfer1::DataType type) {
    //     switch (type) {
    //         case nvinfer1::DataType::kFLOAT:
    //             return "kFLOAT (32-bit float)";
    //         case nvinfer1::DataType::kHALF:
    //             return "kHALF (16-bit float)";
    //         case nvinfer1::DataType::kINT8:
    //             return "kINT8 (8-bit integer)";
    //         case nvinfer1::DataType::kBOOL:
    //             return "kBOOL (boolean)";
    //         case nvinfer1::DataType::kUINT8:
    //             return "kUINT8 (8-bit unsigned integer)";
    //         default:
    //             return "Unknown DataType";
    //     }
    // }

    void init(std::shared_ptr<nvinfer1::ICudaEngine> const engine, const int batchSize = 0, std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr)
    {
        mEngine = engine ;
        mBatchSize = batchSize ;

        mBatchSize = batchSize ; 
        assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);

        for (int i = 0; i < mEngine->getNbIOTensors(); i++)
        {

            const char * binding_name = engine->getIOTensorName(i);
            auto dims = context ? context->getTensorShape(binding_name) : mEngine->getTensorShape(binding_name);
            size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getTensorDataType(binding_name);
            int vecDim = mEngine->getTensorVectorizedDim(binding_name);
            vol *= volume(dims);

            // std::cout << " BuferManager Init, binding_name " << binding_name <<", type" << dataTypeToString(type)  << ", vol" << vol << std::endl;
            
            std::unique_ptr<ManagedBuffer> manBuf =  std::make_unique<ManagedBuffer>();
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type); 
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf));

        }
    }


    // void init(std::shared_ptr<nvinfer1::ICudaEngine> const engine, const int batchSize = 0,const nvinfer1::IExecutionContext* context = nullptr)
    // {
    //     mEngine = engine ;
    //     mBatchSize = batchSize ;

    //     mBatchSize = batchSize ; 
    //     assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);

    //     for (int i = 0; i < mEngine->getNbBindings(); i++)
    //     {

    //         auto dims = context ? context->getBindingShape(i) : mEngine->getBindingShape(i);
    //         size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
    //         nvinfer1::DataType type = mEngine->getBindingDataType(i);
    //         int vecDim = mEngine->getBindingVectorizedDim(i);
    //         vol *= volume(dims);
            
    //         std::unique_ptr<ManagedBuffer> manBuf =  std::make_unique<ManagedBuffer>();
    //         manBuf->deviceBuffer = DeviceBuffer(vol, type);
    //         manBuf->hostBuffer = HostBuffer(vol, type); 
    //         mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
    //         mManagedBuffers.emplace_back(std::move(manBuf));

    //     }
    // }


    /**
     * @brief Create a BufferManager for handling buffer interactions with engine.
     * 
     * @param engine 
     * @param batchSize 
     * @param context 
     */
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batchSize = 0,const nvinfer1::IExecutionContext* context = nullptr)
        : mEngine(engine)
        , mBatchSize(batchSize)
    {
        // Full Dims implies no batch size.
        assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
        // Create host and device buffers
        for (int i = 0; i < mEngine->getNbIOTensors(); i++)
        {

            const char * binding_name = engine->getIOTensorName(i);

            auto dims = context ? context->getTensorShape(binding_name) : mEngine->getTensorShape(binding_name);
            size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getTensorDataType(binding_name);
            int vecDim = mEngine->getTensorVectorizedDim(binding_name);
            vol *= volume(dims);

            // if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
            // {
            //     int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            //     dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
            //     vol *= scalarsPerVec;
            // }
            
            std::unique_ptr<ManagedBuffer> manBuf =  std::make_unique<ManagedBuffer>();
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type); 
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf));

        }
    }


    /**
     * @brief Returns a vector of device buffers that you can use directly as
     *        bindings for the execute and enqueue methods of IExecutionContext.
     * @return std::vector<void*>& 
     */
    std::vector<void*>& getDeviceBindings()
    {
        return mDeviceBindings;
    }

    /**
     * @brief  Returns a vector of device buffers.
     * 
     * @return const std::vector<void*>& 
     */
    const std::vector<void*>& getDeviceBindings() const
    {
        return mDeviceBindings;
    }


    /**
     * @brief Returns the device buffer corresponding to tensorName.
     *        Returns nullptr if no such tensor can be found.
     * 
     * @param tensorName 
     * @return void* 
     */
    void* getDeviceBuffer(int index) const
    {
        return getBuffer(false, index);
    }

    /**
     * @brief Returns the host buffer corresponding to tensorName.
     *        Returns nullptr if no such tensor can be found.
     * 
     * @param tensorName 
     * @return void* 
     */
    void* getHostBuffer(int index) const
    {
        return getBuffer(true, index);
    }

    /**
     * @brief Returns the size of the host and device buffers that correspond to tensorName.
     *  Returns kINVALID_SIZE_VALUE if no such tensor can be found.
     * 
     * @param tensorName 
     * @return size_t 
     */
    size_t size(const std::string& tensorName) const
    {
        return mEngine->getTensorBytesPerComponent(tensorName.c_str());
        // int index = mEngine->getBindingIndex(tensorName.c_str());
        // if (index == -1)
        //     return kINVALID_SIZE_VALUE;
        // return mManagedBuffers[index]->hostBuffer.nbBytes();
    }


    /**
     * @brief Templated print function that dumps buffers of arbitrary type to std::ostream.
     *        rowCount parameter controls how many elements are on each line.
     *        A rowCount of 1 means that there is only 1 element on each line.
     * 
     * @tparam T 
     * @param os 
     * @param buf 
     * @param bufSize 
     * @param rowCount 
     */
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    /**
     * @brief Copy the contents of input host buffers to input device buffers synchronously.
     * 
     */
    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    /**
     * @brief Copy the contents of output device buffers to output host buffers synchronously.
     * 
     */
    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }


    /**
     * @brief Copy the contents of input host buffers to input device buffers asynchronously.
     * 
     * @param stream 
     */
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }



    /**
     * @brief Copy the contents of output device buffers to output host buffers asynchronously.
     * 
     * @param stream 
     */
    void copyOutputToHostAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager()
    {
        std::cout<< "~BufferManager" << std::endl;
    }

    // ~BufferManager() = default;

    // ~BufferManager()
    // {
    //     // std::cout<< "~BufferManager" << std::endl ;
    // } 




private:
    void* getBuffer(const bool isHost, int index) const
    {
        return (isHost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());

    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
    {
        for (int i = 0; i < mEngine->getNbIOTensors(); i++)
        {
            void* dstPtr = deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffer.data();
            const void* srcPtr = deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffer.data();
            const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            const char * binding_name = mEngine->getIOTensorName(i);
            bool input;
            if(mEngine->getTensorIOMode(binding_name) ==  nvinfer1::TensorIOMode::kINPUT)
            {
                input = true;
            }
            if ((copyInput && input) || (!copyInput && !input))
            {
                if (async)
                    cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
                else
                    cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              
    int mBatchSize;                                              
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; 
    std::vector<void*> mDeviceBindings;                          
};

}

#endif 

