namespace HRMS.Models.Requests;

public class UpdateLocationRequest: LocationDetails
{
    public int ModifiedBy { get; set; }
}