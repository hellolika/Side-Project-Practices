using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces;


public interface IImageService
{
    public Task<ApiBaseResponse<UploadImageResponse>> UploadImage(UploadImageRequest request);
    Task<bool> CheckImageSize(IFormFile request);
}