CREATE PROCEDURE [dbo].[HRMS_AddAnnouncement.1.0.0]
	@title AS NVARCHAR(200),
	@message AS NVARCHAR(2000),
    @createdBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

	INSERT INTO [dbo].[Announcement]
	([Title], [Message], [CreatedBy], [ModifiedBy])
	VALUES(@title, @message, @createdBy, @createdBy);

	SELECT 0 AS ErrorCode;
END