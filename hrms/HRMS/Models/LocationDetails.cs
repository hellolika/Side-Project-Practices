using Newtonsoft.Json;

namespace HRMS.Models
{
    public class LocationDetails
    {
        [JsonProperty("Id")]
        public virtual int Id { get; set; }

        [JsonProperty("LocationName")]
        public string LocationName { get; set; }

        [JsonProperty("Latitude")]
        public string Latitude { get; set; }

        [JsonProperty("Longitude")]
        public string Longitude { get; set; }

        [JsonProperty("Range")]
        public int Range { get; set; }

        [JsonProperty("IsEnabled")]
        public virtual bool IsEnabled { get; set; }
    }
}