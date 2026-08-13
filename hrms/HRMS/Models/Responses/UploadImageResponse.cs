using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class UploadImageResponse
    {
        [JsonProperty("ImagePath")]
        public string ImagePath { get; set; }
    }
}

