#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            int id = frameInfo.m_id(x, y);
            if(id == -1)
                continue;
            
            Matrix4x4 world_to_local = Inverse(frameInfo.m_matrix[id]);
            const Matrix4x4 &pre_local_to_world = m_preFrameInfo.m_matrix[id];
            auto position = frameInfo.m_position(x, y);
            //代码中一共做了3步变换，其中world_to_local对应 M_i^-1 ，pre_local_to_world对应 M_i-1，pre_World_To_Screen对应 P_i-1 * V_i-1。
            // 相当于下面这一行，但是内部嵌套太多，无法顺利运行，原因不明  
            //auto screen_position = preWorldToScreen(pre_local_to_world(world_to_local(position, Float3::EType::Point), Float3::EType::Point) , Float3::EType::Point)

            auto world_position     = frameInfo.m_position(x, y);
            auto pre_local_position = world_to_local(world_position, Float3::EType::Point);
            auto pre_world_position = pre_local_to_world(pre_local_position, Float3::EType::Point);
            auto pre_screen_position    = preWorldToScreen(pre_world_position, Float3::EType::Point);

            if (pre_screen_position.x < 0 || pre_screen_position.x >= width ||
                pre_screen_position.y < 0 || pre_screen_position.y >= height )
                continue;
            else{
                int pre_id = m_preFrameInfo.m_id(pre_screen_position.x, pre_screen_position.y);
                if(pre_id == id){
                    m_valid(x, y) = true;
                    m_misc(x, y)  = m_accColor(pre_screen_position.x, pre_screen_position.y);
                }
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 7;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            // TODO: Exponential moving average
            float alpha = 1.0f;

            if (m_valid(x, y)){
                alpha = m_alpha;

                int x_start = std::max(x - kernelRadius, 0);
                int x_end   = std::min(x + kernelRadius, width - 1);
                int y_start = std::max(y - kernelRadius, 0);
                int y_end   = std::min(y + kernelRadius, height - 1);

                Float3 mu(0.f);
                Float3 sigma(0.f);

                for (int n = y_start; n <= y_end; ++n){
                    for (int m = x_start; m <= x_end; ++m){
                        mu += curFilteredColor(m, n);
                        sigma += Sqr(curFilteredColor(x, y) - curFilteredColor(m, n));
                    }
                }

                int count = kernelRadius * 2 + 1;
                count *= count;
                mu   /= float(count);
                sigma = SafeSqrt(sigma / float(count));
                color = Clamp(color, mu - sigma * m_colorBoxK, mu + sigma * m_colorBoxK);
            }

            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 32;      // 64^2

//是OpenMP中的一个指令，表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系 collapse(2) 两重循环
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            //filteredImage(x, y) = frameInfo.m_beauty(x, y);

            int x_start = std::max(x - kernelRadius, 0);
            int x_end   = std::min(x + kernelRadius, width - 1);
            int y_start = std::max(y - kernelRadius, 0);
            int y_end   = std::min(y + kernelRadius, height - 1);

            auto center_position = frameInfo.m_position(x, y);
            auto center_normal   = frameInfo.m_normal(x, y);
            auto center_color    = frameInfo.m_beauty(x, y);

            Float3 final_color;
            auto total_weight =  0.0f;

            for (int m = x_start; m <= x_end; m++) {
                for (int n = y_start; n <= y_end; n++) {

                    auto postion = frameInfo.m_position(m, n);
                    auto normal  = frameInfo.m_normal(m, n);
                    auto color   = frameInfo.m_beauty(m, n);

                    auto d_position = SqrDistance(center_position, postion) / (2.0f * m_sigmaCoord);
                    auto d_color    = SqrDistance(center_color, color) / (2.0f * m_sigmaColor);
                    auto d_normal   = SafeAcos(Dot(center_normal, normal));
                    d_normal *= d_normal;
                    d_normal /= (2.0f * m_sigmaNormal);

                    float d_plane = .0f;
                    if (d_position > 0.f) { // 防止 Normalize(postion - center_position) 出现问题
                        d_plane = Dot(center_normal, Normalize(postion - center_position));
                    }
                    d_plane *= d_plane;
                    d_plane /= (2.0f * m_sigmaPlane);

                    // d_xx > 0, 各个系数 d_xx 越小，(-d_plane - d_position - d_color - d_normal)越趋近于 -0， weight越趋近于1，也就越大, 符合物理意义
                    float weight = std::exp(-d_plane - d_position - d_color - d_normal);        
                    total_weight += weight;
                    final_color += color * weight;
                }
            }

            filteredImage(x, y) = final_color / total_weight;
        }
    }
    return filteredImage;
}

Buffer2D<Float3> Denoiser::ATrousFilter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int passes = 5;      // 64^2 == 5^2 * 5

//是OpenMP中的一个指令，表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系 collapse(2) 两重循环
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            Float3 final_color;
            auto total_weight =  0.0f;

            for(int pass = 0; pass < passes; ++pass){
                int step = std::pow(2, pass);
                int kernelRadius = step * 2;

                int x_start = std::max(x - kernelRadius, 0);
                int x_end   = std::min(x + kernelRadius, width - 1);
                int y_start = std::max(y - kernelRadius, 0);
                int y_end   = std::min(y + kernelRadius, height - 1);

                auto center_position = frameInfo.m_position(x, y);
                auto center_normal   = frameInfo.m_normal(x, y);
                auto center_color    = frameInfo.m_beauty(x, y);

                for (int n = y_start; n <= y_end; n += step){
                    for (int m = x_start; m <= x_end; m += step){

                        auto postion = frameInfo.m_position(m, n);
                        auto normal  = frameInfo.m_normal(m, n);
                        auto color   = frameInfo.m_beauty(m, n);

                        auto d_position = SqrDistance(center_position, postion) / (2.0f * m_sigmaCoord * m_sigmaCoord);
                        auto d_color    = SqrDistance(center_color, color) / (2.0f * m_sigmaColor * m_sigmaColor);
                        auto d_normal   = SafeAcos(Dot(center_normal, normal));
                        d_normal *= d_normal;
                        d_normal /= (2.0f * m_sigmaNormal * m_sigmaNormal);

                        float d_plane = .0f;
                        if (d_position > 0.f) { // 防止 Normalize(postion - center_position) 出现问题
                            d_plane = Dot(center_normal, Normalize(postion - center_position));
                        }
                        d_plane *= d_plane;
                        d_plane /= (2.0f * m_sigmaPlane * m_sigmaPlane);

                        // d_xx > 0, 各个系数 d_xx 越小，(-d_plane - d_position - d_color - d_normal)越趋近于 -0， weight越趋近于1，也就越大, 符合物理意义
                        float weight = std::exp(-d_plane - d_position - d_color - d_normal);        
                        total_weight += weight;
                        final_color += color * weight;
                    }
                }
            }
            filteredImage(x, y) = final_color / total_weight;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    //filteredColor = Filter(frameInfo);
    filteredColor = ATrousFilter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
