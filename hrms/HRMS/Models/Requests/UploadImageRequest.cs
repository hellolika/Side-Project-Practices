using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class UploadImageRequest
    {
        [JsonProperty("FormFile")]
        public IFormFile FormFile { get; set; }
        [JsonIgnore]
        public string Folder { get; set; }
    
    }
}

