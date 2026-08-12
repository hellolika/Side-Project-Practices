using CloudinaryDotNet;
using CloudinaryDotNet.Actions;
using HRMS.Enum;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Models.Settings;

namespace HRMS.Services.Interfaces;

public class ImageService : IImageService
{
    private readonly CloudinarySettings _cloudinarySettings;
    private readonly ILoggerService _loggerService;

    public ImageService(CloudinarySettings cloudinarySettings, ILoggerService loggerService)
    {
        _cloudinarySettings = cloudinarySettings;
        _loggerService = loggerService;
    }

    public async Task<ApiBaseResponse<UploadImageResponse>> UploadImage(UploadImageRequest request)
    {
        try
        {
            var cloud = _cloudinarySettings.CloudName;
            var apiKey = _cloudinarySettings.ApiKey;
            var apiSecret = _cloudinarySettings.ApiSecret;
            var cloudinary = new Cloudinary(new Account(cloud, apiKey, apiSecret));
            using (var stream = request.FormFile.OpenReadStream())
            {
                var uploadParams = new ImageUploadParams()
                {
                    File = new FileDescription(request.FormFile.FileName, stream),
                    Folder = request.Folder,
                    Transformation = new Transformation().Crop("limit").Width(1000).Height(1000)
                };
                var uploadResult = await cloudinary.UploadAsync(uploadParams);
                var imagePath = uploadResult.PublicId + "." + uploadResult.Format;
                return new ApiBaseResponse<UploadImageResponse>(new UploadImageResponse() { ImagePath = imagePath });
            }

        }
        catch (Exception ex)
        {
            // Log the error
            _loggerService.Error(
                $"[][Ordinary]  Response With Error: Message : {ex.Message} , Stack : {ex.StackTrace}");
            return new ApiBaseResponse<UploadImageResponse>(ApiErrorEnum.UploadImageFail);
        }

    }

    public async Task<bool> CheckImageSize(IFormFile image)
    {
        using (var stream = image.OpenReadStream())
        {
            byte[] buffer = new byte[image.Length];
            await stream.ReadAsync(buffer, 0, (int)image.Length);

            if (buffer.Length > 1024 * 1024)
            {
                return false;
            }
        }

        return true;
    }
}
